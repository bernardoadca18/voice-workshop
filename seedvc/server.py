"""
Serviço HTTP persistente para o Seed-VC.
Carrega o modelo UMA VEZ no startup (load_models) e reaproveita em
todas as conversões -- elimina o custo de recarregar checkpoint/CUDA
a cada chunk, que era a causa real dos timeouts.
"""
import argparse
import logging
import os
import time
from pathlib import Path

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
_modelo_cache: dict[str, tuple] = {}

# cache de conditioning de referência: evita recalcular S_ori, mel2, style2, prompt_condition
# a cada chamada quando target_path/run_name não mudam (que é sempre neste projeto)
_ref_cache: dict[tuple[str, str], tuple] = {}  # (run_name, target_path) -> (mel2, style2, prompt_condition)


def _carregar(run_name: str, fp16: bool = True) -> tuple:
    if run_name not in _modelo_cache:
        logger.info("Carregando modelo para run_name=%s (fp16=%s)...", run_name, fp16)
        t0 = time.time()
        args = argparse.Namespace(
            checkpoint="/data/checkpoints/ft_model.pth",
            config=f"/data/checkpoints/{run_name}.yml",
            fp16=fp16,
            f0_condition=False,
        )
        _modelo_cache[run_name] = load_models(args)
        logger.info("Modelo carregado em %.1fs", time.time() - t0)
    return _modelo_cache[run_name]


def _obter_conditioning_referencia(
    run_name: str,
    target_path: str,
    model,
    semantic_fn,
    campplus_model,
    mel_fn,
    sr: int,
) -> tuple:
    """
    Obtém o conditioning de referência (mel2, style2, prompt_condition) do áudio alvo.
    Usa cache para evitar recalcular a cada chamada quando target_path/run_name não mudam.

    Returns:
        Tupla (mel2, style2, prompt_condition)
    """
    chave = (run_name, target_path)
    if chave in _ref_cache:
        logger.debug("Reutilizando conditioning de referência cacheado para %s", chave)
        return _ref_cache[chave]

    logger.info("Calculando conditioning de referência (1x) para run_name=%s target_path=%s", run_name, target_path)

    ref_audio = librosa.load(target_path, sr=sr)[0]
    ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel2 = mel_fn(ref_audio.to(device).float())
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(
        ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    prompt_condition, _, _, _, _ = model.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=None
    )

    resultado = (mel2, style2, prompt_condition)
    _ref_cache[chave] = resultado
    return resultado


@torch.no_grad()
def converter(
    source_path: str,
    target_path: str,
    output_path: str,
    run_name: str,
    diffusion_steps: int = 50,
    length_adjust: float = 1.0,
    inference_cfg_rate: float = 0.7,
    fp16: bool = True,
) -> str:
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = _carregar(run_name, fp16)
    sr = mel_fn_args["sampling_rate"]

    source_audio = librosa.load(source_path, sr=sr)[0]

    hop_length = 256
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)

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

    # Usa cache de conditioning de referência (evita recálculo redundante)
    mel2, style2, prompt_condition = _obter_conditioning_referencia(
        run_name, target_path, model, semantic_fn, campplus_model, mel_fn, sr
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

    vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
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
    inference_cfg_rate: float = 0.7
    fp16: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "modelos_carregados": list(_modelo_cache.keys())}


@app.post("/cache/limpar")
def limpar_cache():
    """Invalida o cache de conditioning de referência. Útil quando a amostra de referência é trocada sem reiniciar o container."""
    global _ref_cache
    _ref_cache = {}
    logger.info("Cache de conditioning de referência invalidado")
    return {"status": "ok", "mensagem": "Cache de conditioning de referência invalidado"}


@app.post("/convert")
def convert_endpoint(req: ConvertRequest):
    if not Path(req.source_path).exists():
        raise HTTPException(400, f"source_path não existe: {req.source_path}")
    if not Path(req.target_path).exists():
        raise HTTPException(400, f"target_path não existe: {req.target_path}")
    try:
        t0 = time.time()
        resultado = converter(
            req.source_path, req.target_path, req.output_path, req.run_name,
            diffusion_steps=req.diffusion_steps, length_adjust=req.length_adjust,
            inference_cfg_rate=req.inference_cfg_rate, fp16=req.fp16,
        )
        logger.info("Conversão concluída em %.1fs -> %s", time.time() - t0, resultado)
    except Exception as e:
        logger.exception("Falha na conversão")
        raise HTTPException(500, str(e)) from e
    return {"output_path": resultado}