"""
Pipeline Kokoro -> fatiamento por silêncio -> Seed-VC por chunk -> concatenação.

Uso via CLI:
    python3 run_pipeline.py "texto" [run_name]

Uso programático (via api.py):
    from run_pipeline import gerar_narracao
    resultado = gerar_narracao(texto, run_name)
"""
import glob
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

from audio_chunker import fatiar_por_silencio, salvar_chunks, concatenar_com_crossfade

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("run_pipeline")

KOKORO_URL = "http://localhost:8880/v1/audio/speech"
KOKORO_API_KEY = os.environ.get("KOKORO_API_KEY", "")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "pm_alex")

REF_AUDIO_MINHA_VOZ = "/data/reference/voz_referencia.wav"

DATA_DIR = Path("./data")
KOKORO_OUTPUT_HOST = DATA_DIR / "f5_output" / "gerado.wav"
CHUNKS_DIR_HOST = DATA_DIR / "chunks"
CHUNKS_DIR_CONTAINER = "/data/chunks"
CONVERTED_DIR_HOST = DATA_DIR / "chunks_convertidos"
CONVERTED_DIR_CONTAINER = "/data/chunks_convertidos"
FINAL_OUTPUT_HOST = DATA_DIR / "final_output" / "resultado_final.wav"

CHECKPOINT = "/data/checkpoints/ft_model.pth"
DIFFUSION_STEPS = 50


def run_in_container(container: str, cmd: str) -> None:
    full_cmd = ["docker", "exec", container, "bash", "-c", cmd]
    logger.info("[%s] %s", container, cmd)
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Falha no container {container}")


def gerar_audio_kokoro(texto: str, saida_host: Path = KOKORO_OUTPUT_HOST) -> Path:
    """Etapa 1: gera o áudio base contínuo, sem fatiar o texto."""
    headers = {"Content-Type": "application/json"}
    if KOKORO_API_KEY:
        headers["Authorization"] = f"Bearer {KOKORO_API_KEY}"

    resp = requests.post(
        KOKORO_URL,
        headers=headers,
        json={"model": "tts-1", "input": texto, "voice": KOKORO_VOICE, "response_format": "wav"},
        timeout=120,
    )
    resp.raise_for_status()

    saida_host.parent.mkdir(parents=True, exist_ok=True)
    saida_host.write_bytes(resp.content)
    logger.info("kokoro: áudio base salvo em %s (%d bytes)", saida_host, len(resp.content))
    return saida_host


def converter_chunk_seedvc(
    chunk_path_container: str, run_name: str, indice: int
) -> Path:
    """Roda o Seed-VC para um único chunk e retorna o path do arquivo convertido,
    já renomeado deterministicamente (part_XXX.wav) para preservar a ordem
    na concatenação -- o Seed-VC gera nomes automáticos que dependem dos
    parâmetros usados, então capturamos o mais recente logo após a chamada."""
    config = f"/data/checkpoints/{run_name}.yml"
    antes = set(glob.glob(f"{CONVERTED_DIR_HOST}/vc_*.wav"))

    cmd = (
        f'python3 inference.py '
        f'--source "{chunk_path_container}" '
        f'--target "{REF_AUDIO_MINHA_VOZ}" '
        f'--output "{CONVERTED_DIR_CONTAINER}" '
        f'--diffusion-steps {DIFFUSION_STEPS} '
        f'--length-adjust 1.0 '
        f'--inference-cfg-rate 0.7 '
        f'--f0-condition False '
        f'--fp16 True '
        f'--checkpoint "{CHECKPOINT}" '
        f'--config "{config}"'
    )
    run_in_container("seedvc", cmd)

    depois = set(glob.glob(f"{CONVERTED_DIR_HOST}/vc_*.wav"))
    novos = depois - antes
    if not novos:
        raise RuntimeError(f"Seed-VC não gerou saída nova para o chunk {indice}")
    gerado = Path(max(novos, key=lambda p: os.path.getmtime(p)))

    renomeado = CONVERTED_DIR_HOST / f"part_{indice:03d}.wav"
    gerado.rename(renomeado)
    return renomeado


def gerar_narracao(texto: str, run_name: str = "minha_voz_v1") -> Path:
    """
    Função principal, ponta a ponta. É o que a API chama.

    1. Kokoro gera o bloco contínuo (prosódia intacta).
    2. Fatiamento por silêncio real do áudio já gerado (não do texto).
    3. Cada chunk passa pelo Seed-VC individualmente (janela 4-8s validada).
    4. Concatenação com crossfade.
    """
    t0 = time.time()

    audio_base = gerar_audio_kokoro(texto)

    chunks = fatiar_por_silencio(audio_base)
    logger.info("fatiamento: %d chunk(s) gerado(s) a partir do áudio contínuo", len(chunks))
    chunk_paths_host = salvar_chunks(chunks, CHUNKS_DIR_HOST)

    CONVERTED_DIR_HOST.mkdir(parents=True, exist_ok=True)
    convertidos = []
    for i, chunk_path in enumerate(chunk_paths_host, start=1):
        nome_arquivo = chunk_path.name
        chunk_path_container = f"{CHUNKS_DIR_CONTAINER}/{nome_arquivo}"
        convertido = converter_chunk_seedvc(chunk_path_container, run_name, i)
        convertidos.append(convertido)
        logger.info("seed-vc: chunk %d/%d convertido -> %s", i, len(chunk_paths_host), convertido)

    resultado = concatenar_com_crossfade(convertidos, FINAL_OUTPUT_HOST)

    logger.info("pipeline concluído em %.1fs -> %s", time.time() - t0, resultado)
    return resultado


if __name__ == "__main__":
    texto_cli = sys.argv[1] if len(sys.argv) > 1 else "Texto de teste para gerar o áudio."
    run_name_cli = sys.argv[2] if len(sys.argv) > 2 else "minha_voz_v1"

    if not KOKORO_API_KEY:
        logger.warning("KOKORO_API_KEY não definida -- a chamada falha com 401 se a API exigir auth")

    resultado = gerar_narracao(texto_cli, run_name_cli)
    print(f"Pronto! Resultado em {resultado}")