"""
Serviço HTTP persistente para o Seed-VC.
Mantém somente um modelo ativo por padrão e o reaproveita entre conversões,
com serialização de carga/inferência e limpeza de memória entre chamadas.
"""
import argparse
import ctypes
import gc
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

# Evita explosão de threads BLAS/OpenMP durante pré-processamento sob pressão.
CPU_THREADS = max(1, int(os.environ.get("SEEDVC_CPU_THREADS", "2")))
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import librosa
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import load_models, crossfade, adjust_f0_semitones, device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seedvc-server")

app = FastAPI(title="Seed-VC Service")

# cache por run_name -- troca de voz não recarrega do zero se já foi
# usada nesta sessão, mas o caso comum é 1 run_name só, carregado 1x
_modelo_cache: OrderedDict[tuple[str, bool], tuple] = OrderedDict()

# cache de conditioning de referência: evita recalcular S_ori, mel2, style2, prompt_condition
# a cada chamada quando target_path/run_name não mudam (que é sempre neste projeto)
_ref_cache: dict[tuple[tuple[str, bool], str, int, int], tuple] = {}


try:
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
except RuntimeError:
    logger.warning("Não foi possível ajustar threads do PyTorch; mantendo configuração atual")

MAX_CACHED_MODELS = max(1, int(os.environ.get("SEEDVC_MAX_CACHED_MODELS", "1")))
INFERENCE_LOCK_TIMEOUT_S = float(os.environ.get("SEEDVC_INFERENCE_LOCK_TIMEOUT_S", "900"))

# Endpoints sync do FastAPI rodam em um thread pool. Este RLock cobre load e
# inferência, impedindo cargas simultâneas do mesmo modelo.
_resource_lock = threading.RLock()


def _memory_snapshot() -> dict[str, float]:
    """Telemetria barata para logs/health sem dependências adicionais."""
    snapshot: dict[str, float] = {}
    try:
        status = Path("/proc/self/status").read_text()
        for line in status.splitlines():
            if line.startswith(("VmRSS:", "VmSwap:")):
                key, value, *_ = line.split()
                snapshot[key.rstrip(":").lower() + "_mb"] = round(int(value) / 1024, 1)
    except OSError:
        pass
    if torch.cuda.is_available():
        snapshot["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
        snapshot["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 1)
    return snapshot


def _trim_process_memory() -> None:
    """Libera caches CUDA transitórios e arenas livres do glibc ao SO."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except (OSError, AttributeError):
        logger.debug("malloc_trim indisponível nesta plataforma")


def _cleanup_runtime(reason: str) -> None:
    _trim_process_memory()
    logger.info("Recursos após %s: %s", reason, _memory_snapshot())


def _drop_reference_cache_for(model_key: tuple[str, bool]) -> None:
    for key in [key for key in _ref_cache if key[0] == model_key]:
        del _ref_cache[key]


def _evict_models_if_needed() -> None:
    while len(_modelo_cache) >= MAX_CACHED_MODELS:
        old_key, old_bundle = _modelo_cache.popitem(last=False)
        _drop_reference_cache_for(old_key)
        logger.info("Descarregando modelo LRU: run_name=%s fp16=%s", *old_key)
        del old_bundle
        _cleanup_runtime(f"descarregar modelo {old_key[0]}")



def _carregar(run_name: str, fp16: bool = True) -> tuple:
    model_key = (run_name, bool(fp16))
    with _resource_lock:
        cached = _modelo_cache.get(model_key)
        if cached is not None:
            _modelo_cache.move_to_end(model_key)
            return cached

        _evict_models_if_needed()
        logger.info("Carregando modelo para run_name=%s (fp16=%s)...", run_name, fp16)
        t0 = time.time()

        if run_name == "zero_shot":
            args = argparse.Namespace(
                checkpoint=None,
                config=None,
                fp16=fp16,
                f0_condition=False,
            )
        else:
            checkpoint_path = f"/data/checkpoints/{run_name}.pth"
            config_path = f"/data/checkpoints/{run_name}.yml"
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
            if not Path(config_path).exists():
                raise FileNotFoundError(f"Config não encontrado: {config_path}")
            args = argparse.Namespace(
                checkpoint=checkpoint_path,
                config=config_path,
                fp16=fp16,
                f0_condition=False,
            )

        try:
            bundle = load_models(args)
        except Exception:
            _cleanup_runtime(f"falha ao carregar modelo {run_name}")
            raise

        _modelo_cache[model_key] = bundle
        logger.info("Modelo carregado em %.1fs", time.time() - t0)
        _cleanup_runtime(f"carregar modelo {run_name}")
        return bundle

def _obter_conditioning_referencia(
    run_name: str,
    target_path: str,
    model,
    semantic_fn,
    campplus_model,
    mel_fn,
    sr: int,
    fp16: bool,
) -> tuple:
    """Obtém conditioning cacheado e o invalida se a referência mudar."""
    model_key = (run_name, bool(fp16))
    resolved_target = str(Path(target_path).resolve())
    stat = Path(target_path).stat()
    chave = (model_key, resolved_target, stat.st_mtime_ns, stat.st_size)
    if chave in _ref_cache:
        logger.debug("Reutilizando conditioning de referência cacheado para %s", chave)
        return _ref_cache[chave]

    # Remove fingerprint antiga do mesmo arquivo/modelo.
    for old_key in [
        key for key in _ref_cache
        if key[0] == model_key and key[1] == resolved_target
    ]:
        del _ref_cache[old_key]

    logger.info(
        "Calculando conditioning de referência para run_name=%s target_path=%s",
        run_name,
        target_path,
    )
    ref_samples = librosa.load(target_path, sr=sr)[0]
    ref_audio = torch.from_numpy(
        np.ascontiguousarray(ref_samples[:sr * 25])
    ).unsqueeze(0).float().to(device)
    del ref_samples

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)
    mel2 = mel_fn(ref_audio)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(
        ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))
    prompt_condition, _, _, _, _ = model.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
    )

    resultado = tuple(
        tensor.detach() for tensor in (mel2, style2, prompt_condition)
    )
    _ref_cache[chave] = resultado
    return resultado


@torch.inference_mode()
def converter(
    source_path: str,
    target_path: str,
    output_path: str,
    run_name: str,
    diffusion_steps: int = 50,
    length_adjust: float = 1.0,
    inference_cfg_rate: float = 0.85,
    fp16: bool = True,
) -> str:
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = _carregar(run_name, fp16)
    sr = mel_fn_args["sampling_rate"]

    source_samples = librosa.load(source_path, sr=sr)[0]

    hop_length = 256
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    source_audio = torch.from_numpy(
        np.ascontiguousarray(source_samples)
    ).unsqueeze(0).float().to(device)

    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:
                chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]],
                    dim=-1)
            S_alt_chunk = semantic_fn(chunk)
            S_alt_list.append(S_alt_chunk if traversed_time == 0 else S_alt_chunk[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    del source_samples
    mel2, style2, prompt_condition = _obter_conditioning_referencia(
        run_name, target_path, model, semantic_fn, campplus_model, mel_fn, sr, fp16
    )

    mel = mel_fn(source_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)

    cond, _, codes, _, _ = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)

    max_source_window = max_context_window - mel2.size(2)
    processed_frames = 0
    generated_wave_chunks = []
    previous_chunk = None

    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
            vc_target = model.cfm.inference(
                cat_condition, torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2, style2, None, diffusion_steps, inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = vocoder_fn(vc_target.float()).squeeze()[None, :]

        if processed_frames == 0:
            if is_last_chunk:
                generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                break
            generated_wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            generated_wave_chunks.append(crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len))
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            generated_wave_chunks.append(crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len))
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    output_samples = np.ascontiguousarray(np.concatenate(generated_wave_chunks))
    vc_wave = torch.from_numpy(output_samples)[None, :].float()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_path, vc_wave.cpu(), sr)
    return output_path


class ConvertRequest(BaseModel):
    source_path: str
    target_path: str
    output_path: str
    run_name: str = "minha_voz_v1"
    diffusion_steps: int = 50
    length_adjust: float = 1.0
    inference_cfg_rate: float = 0.85
    fp16: bool = True


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelos_carregados": [key[0] for key in _modelo_cache],
        "max_modelos_em_cache": MAX_CACHED_MODELS,
        "recursos": _memory_snapshot(),
    }


@app.post("/cache/limpar")
def limpar_cache():
    """Invalida conditioning de referência e devolve memória livre ao SO."""
    with _resource_lock:
        _ref_cache.clear()
        _cleanup_runtime("limpar cache de referência")
    return {"status": "ok", "mensagem": "Cache de conditioning invalidado"}


@app.post("/convert")
def convert_endpoint(req: ConvertRequest):
    if not Path(req.source_path).exists():
        raise HTTPException(400, f"source_path não existe: {req.source_path}")
    if not Path(req.target_path).exists():
        raise HTTPException(400, f"target_path não existe: {req.target_path}")

    acquired = _resource_lock.acquire(timeout=INFERENCE_LOCK_TIMEOUT_S)
    if not acquired:
        raise HTTPException(503, "Seed-VC ocupado; timeout aguardando slot de inferência")

    t0 = time.time()
    try:
        resultado = converter(
            req.source_path, req.target_path, req.output_path, req.run_name,
            diffusion_steps=req.diffusion_steps, length_adjust=req.length_adjust,
            inference_cfg_rate=req.inference_cfg_rate, fp16=req.fp16,
        )
        logger.info("Conversão concluída em %.1fs -> %s", time.time() - t0, resultado)
        return {"output_path": resultado, "recursos": _memory_snapshot()}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Falha na conversão")
        raise HTTPException(500, str(e)) from e
    finally:
        try:
            _cleanup_runtime(f"conversão run_name={req.run_name}")
        finally:
            # Cleanup nunca pode deixar o serviço permanentemente bloqueado.
            _resource_lock.release()