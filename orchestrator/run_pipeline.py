"""
Pipeline Kokoro -> fatiamento por silêncio -> Seed-VC (HTTP) por chunk -> concatenação.
"""
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from audio_chunker import fatiar_por_silencio, salvar_chunks, concatenar_com_crossfade
from audio_postprocess import normalizar_loudness
from glossario_fonetico import aplicar_respelling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("run_pipeline")


_http_session = requests.Session()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    # ReadTimeout não é seguro para retry: o servidor pode continuar
    # processando a requisição original. Retentamos apenas falha de conexão.
    retry=retry_if_exception_type(requests.ConnectionError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _post_com_retry(url: str, **kwargs):
    """POST para operações idempotentes antes de iniciarem processamento."""
    resp = _http_session.post(url, **kwargs)
    resp.raise_for_status()
    return resp


KOKORO_URL = os.environ.get("KOKORO_URL", "http://localhost:8880/v1/audio/speech")
KOKORO_API_KEY = os.environ.get("KOKORO_API_KEY", "")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "pm_alex")

SEEDVC_URL = os.environ.get("SEEDVC_URL", "http://localhost:8001")

REF_AUDIO_MINHA_VOZ = "/data/reference/voz_referencia.wav"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINTS_DIR_HOST = DATA_DIR / "checkpoints"
FINAL_OUTPUT_DIR_HOST = DATA_DIR / "final_output"

DIFFUSION_STEPS_DRAFT = int(os.environ.get("SEEDVC_DIFFUSION_STEPS_DRAFT", "35"))
DIFFUSION_STEPS_FINAL = int(os.environ.get("SEEDVC_DIFFUSION_STEPS_FINAL", "75"))
SEEDVC_TIMEOUT_S = int(os.environ.get("SEEDVC_TIMEOUT_S", "600"))


class RunNameInvalido(Exception):
    """Levantado quando o run_name não tem checkpoint/config treinados."""


def validar_run_name(run_name: str) -> None:
    """
    Valida se o run_name possui checkpoint e config treinados.
    zero_shot é tratado separadamente (não requer checkpoint).
    """
    if run_name == "zero_shot":
        # zero_shot usa modelo pretrained do Seed-VC, não requer checkpoint local
        return

    checkpoint = CHECKPOINTS_DIR_HOST / f"{run_name}.pth"
    config = CHECKPOINTS_DIR_HOST / f"{run_name}.yml"

    if not checkpoint.exists():
        raise RunNameInvalido(
            f"Checkpoint não encontrado: {checkpoint}\n"
            f"Certifique-se de treinar o modelo '{run_name}' antes de usá-lo na inferência."
        )
    if not config.exists():
        raise RunNameInvalido(
            f"Config não encontrado para run_name='{run_name}': {config}\n"
            f"Certifique-se de treinar o modelo '{run_name}' antes de usá-lo na inferência."
        )


def gerar_audio_kokoro(texto: str, saida_host: Path) -> Path:
    """Etapa 1: gera o áudio base contínuo. Aplica respelling fonético
    apenas na cópia enviada ao Kokoro -- o `texto` original nunca é alterado."""
    texto_para_sintese = aplicar_respelling(texto)
    if texto_para_sintese != texto:
        logger.info("kokoro: respelling fonético aplicado para síntese")

    headers = {"Content-Type": "application/json"}
    if KOKORO_API_KEY:
        headers["Authorization"] = f"Bearer {KOKORO_API_KEY}"

    resp = _post_com_retry(
        KOKORO_URL,
        headers=headers,
        json={"model": "tts-1", "input": texto_para_sintese, "voice": KOKORO_VOICE, "response_format": "wav"},
        timeout=120,
    )

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
    diffusion_steps: int,
) -> Path:
    """Converte um chunk chamando o serviço HTTP persistente do Seed-VC
    (modelo já carregado em memória -- sem custo de reload por chunk)."""
    saida_container = f"{converted_dir_container}/part_{indice:03d}.wav"

    # Não há retry automático de ReadTimeout. O Seed-VC pode continuar
    # executando mesmo que o cliente desista; reenviar criaria inferências
    # duplicadas e poderia carregar o mesmo modelo concorrentemente.
    resp = _http_session.post(
        f"{SEEDVC_URL}/convert",
        json={
            "source_path": chunk_path_container,
            "target_path": REF_AUDIO_MINHA_VOZ,
            "output_path": saida_container,
            "run_name": run_name,
            "diffusion_steps": diffusion_steps,
        },
        timeout=(10, SEEDVC_TIMEOUT_S),
    )
    resp.raise_for_status()

    output_path = converted_dir_host / f"part_{indice:03d}.wav"
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Seed-VC respondeu sem produzir output válido: {output_path}")
    return output_path


def gerar_narracao(
    texto: str,
    run_name: str = "minha_voz_v1",
    manter_arquivos_intermediarios: bool = False,
    qualidade: str = "final",
) -> Path:
    """
    Ponta a ponta. Cada chamada usa um diretório de trabalho isolado
    (via request_id) para não colidir com outras chamadas e para
    permitir depuração pontual sem lixo acumulando entre execuções.

    Args:
        texto: Texto a ser narrado.
        run_name: Nome do modelo/voz treinada.
        manter_arquivos_intermediarios: Se True, não remove arquivos temporários.
        qualidade: "draft" (25 passos, rápido) ou "final" (45 passos, melhor qualidade).
    """
    # Mapeia qualidade para diffusion_steps
    if qualidade == "draft":
        diffusion_steps = DIFFUSION_STEPS_DRAFT
    elif qualidade == "final":
        diffusion_steps = DIFFUSION_STEPS_FINAL
    else:
        raise ValueError(f"Qualidade inválida: {qualidade}. Use 'draft' ou 'final'.")

    logger.info("gerar_narracao: qualidade=%s diffusion_steps=%d", qualidade, diffusion_steps)
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
                chunk_path_container, run_name, i, converted_dir_host, converted_dir_container, diffusion_steps
            )
            convertidos.append(convertido)
            logger.info("seed-vc: chunk %d/%d convertido", i, len(chunk_paths_host))

        FINAL_OUTPUT_DIR_HOST.mkdir(parents=True, exist_ok=True)
        resultado_final = FINAL_OUTPUT_DIR_HOST / f"resultado_{request_id}.wav"
        concatenar_com_crossfade(convertidos, resultado_final)

        # Normalização de loudness final (LUFS)
        normalizar_loudness(resultado_final)

        logger.info("gerar_narracao: concluído em %.1fs -> %s", time.time() - t0, resultado_final)
        return resultado_final

    finally:
        if not manter_arquivos_intermediarios and work_dir_host.exists():
            shutil.rmtree(work_dir_host, ignore_errors=True)