"""
Pipeline Kokoro -> fatiamento por silêncio -> Seed-VC por chunk -> concatenação.
"""
import glob
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

from audio_chunker import fatiar_por_silencio, salvar_chunks, concatenar_com_crossfade
from glossario_fonetico import aplicar_respelling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("run_pipeline")

KOKORO_URL = "http://localhost:8880/v1/audio/speech"
KOKORO_API_KEY = os.environ.get("KOKORO_API_KEY", "")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "pm_alex")

REF_AUDIO_MINHA_VOZ = "/data/reference/voz_referencia.wav"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR_HOST = DATA_DIR / "checkpoints"
FINAL_OUTPUT_DIR_HOST = DATA_DIR / "final_output"

DIFFUSION_STEPS = 50
SEEDVC_TIMEOUT_S = 300  # não deixa a chamada pendurada indefinidamente


class RunNameInvalido(Exception):
    """Levantado quando o run_name não tem checkpoint/config treinados."""


def validar_run_name(run_name: str) -> None:
    checkpoint = CHECKPOINTS_DIR_HOST / "ft_model.pth"
    config = CHECKPOINTS_DIR_HOST / f"{run_name}.yml"
    if not checkpoint.exists():
        raise RunNameInvalido(f"Checkpoint não encontrado: {checkpoint}")
    if not config.exists():
        raise RunNameInvalido(f"Config não encontrado para run_name='{run_name}': {config}")


def run_in_container(container: str, cmd: str, timeout: int = SEEDVC_TIMEOUT_S) -> None:
    full_cmd = ["docker", "exec", container, "bash", "-c", cmd]
    logger.info("[%s] %s", container, cmd)
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Timeout ({timeout}s) executando comando em {container}") from e

    if result.stdout:
        logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Falha no container {container}: {result.stderr[:500]}")


def gerar_audio_kokoro(texto: str, saida_host: Path) -> Path:
    """Etapa 1: gera o áudio base contínuo. Aplica respelling fonético
    apenas na cópia enviada ao Kokoro -- o `texto` original nunca é alterado."""
    texto_para_sintese = aplicar_respelling(texto)
    if texto_para_sintese != texto:
        logger.info("kokoro: respelling fonético aplicado para síntese")

    headers = {"Content-Type": "application/json"}
    if KOKORO_API_KEY:
        headers["Authorization"] = f"Bearer {KOKORO_API_KEY}"

    resp = requests.post(
        KOKORO_URL,
        headers=headers,
        json={"model": "tts-1", "input": texto_para_sintese, "voice": KOKORO_VOICE, "response_format": "wav"},
        timeout=120,
    )
    resp.raise_for_status()

    saida_host.parent.mkdir(parents=True, exist_ok=True)
    saida_host.write_bytes(resp.content)
    logger.info("kokoro: áudio base salvo em %s (%d bytes)", saida_host, len(resp.content))
    return saida_host


def converter_chunk_seedvc(
    chunk_path_container: str,
    run_name: str,
    indice: int,
    converted_dir_host: Path,
    converted_dir_container: str,
) -> Path:
    config = f"/data/checkpoints/{run_name}.yml"
    antes = set(glob.glob(f"{converted_dir_host}/vc_*.wav"))

    cmd = (
        f'python3 inference.py '
        f'--source "{chunk_path_container}" '
        f'--target "{REF_AUDIO_MINHA_VOZ}" '
        f'--output "{converted_dir_container}" '
        f'--diffusion-steps {DIFFUSION_STEPS} '
        f'--length-adjust 1.0 '
        f'--inference-cfg-rate 0.7 '
        f'--f0-condition False '
        f'--fp16 True '
        f'--checkpoint "/data/checkpoints/ft_model.pth" '
        f'--config "{config}"'
    )
    run_in_container("seedvc", cmd)

    depois = set(glob.glob(f"{converted_dir_host}/vc_*.wav"))
    novos = depois - antes
    if not novos:
        raise RuntimeError(f"Seed-VC não gerou saída nova para o chunk {indice}")
    gerado = Path(max(novos, key=lambda p: os.path.getmtime(p)))

    renomeado = converted_dir_host / f"part_{indice:03d}.wav"
    gerado.rename(renomeado)
    return renomeado


def gerar_narracao(
    texto: str, run_name: str = "minha_voz_v1", manter_arquivos_intermediarios: bool = False
) -> Path:
    """
    Ponta a ponta. Cada chamada usa um diretório de trabalho isolado
    (via request_id) para não colidir com outras chamadas e para
    permitir depuração pontual sem lixo acumulando entre execuções.
    """
    validar_run_name(run_name)

    request_id = uuid.uuid4().hex[:8]
    work_dir_host = DATA_DIR / "work" / request_id
    work_dir_container = f"/data/work/{request_id}"

    chunks_dir_host = work_dir_host / "chunks"
    chunks_dir_container = f"{work_dir_container}/chunks"
    converted_dir_host = work_dir_host / "convertidos"
    converted_dir_container = f"{work_dir_container}/convertidos"

    t0 = time.time()
    logger.info("gerar_narracao: iniciando request_id=%s", request_id)

    try:
        audio_base = gerar_audio_kokoro(texto, work_dir_host / "gerado.wav")

        chunks = fatiar_por_silencio(audio_base)
        logger.info("fatiamento: %d chunk(s)", len(chunks))
        chunk_paths_host = salvar_chunks(chunks, chunks_dir_host)

        converted_dir_host.mkdir(parents=True, exist_ok=True)
        convertidos = []
        for i, chunk_path in enumerate(chunk_paths_host, start=1):
            chunk_path_container = f"{chunks_dir_container}/{chunk_path.name}"
            convertido = converter_chunk_seedvc(
                chunk_path_container, run_name, i, converted_dir_host, converted_dir_container
            )
            convertidos.append(convertido)
            logger.info("seed-vc: chunk %d/%d convertido", i, len(chunk_paths_host))

        FINAL_OUTPUT_DIR_HOST.mkdir(parents=True, exist_ok=True)
        resultado_final = FINAL_OUTPUT_DIR_HOST / f"resultado_{request_id}.wav"
        concatenar_com_crossfade(convertidos, resultado_final)

        logger.info("gerar_narracao: concluído em %.1fs -> %s", time.time() - t0, resultado_final)
        return resultado_final

    finally:
        if not manter_arquivos_intermediarios and work_dir_host.exists():
            shutil.rmtree(work_dir_host, ignore_errors=True)


if __name__ == "__main__":
    texto_cli = sys.argv[1] if len(sys.argv) > 1 else "Texto de teste para gerar o áudio."
    run_name_cli = sys.argv[2] if len(sys.argv) > 2 else "minha_voz_v1"

    if not KOKORO_API_KEY:
        logger.warning("KOKORO_API_KEY não definida -- a chamada falha com 401 se a API exigir auth")

    resultado = gerar_narracao(texto_cli, run_name_cli, manter_arquivos_intermediarios=True)
    print(f"Pronto! Resultado em {resultado}")