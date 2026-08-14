"""
Pós-processamento de áudio final — normalização de loudness (LUFS).

Usado após a concatenação para garantir que todas as narrações saiam com
o mesmo nível de loudness percebido, adequado para YouTube/streaming.
"""
import logging
import os
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

logger = logging.getLogger("audio_postprocess")

# Loudness alvo em LUFS. -16.0 é um valor comum para narração em YouTube/streaming.
TARGET_LUFS = float(os.environ.get("AUDIO_TARGET_LUFS", "-16.0"))


def normalizar_loudness(caminho_wav: str | Path, target_lufs: float = TARGET_LUFS) -> None:
    """
    Normaliza o loudness integrado (EBU R128) de um arquivo WAV para o alvo especificado.

    Lê o arquivo, mede o loudness integrado, aplica ganho para atingir o alvo,
    e resalva o arquivo no mesmo caminho.

    Args:
        caminho_wav: Caminho para o arquivo WAV.
        target_lufs: Loudness alvo em LUFS. Default: -16.0 (YouTube/streaming).
    """
    caminho_wav = Path(caminho_wav)
    if not caminho_wav.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_wav}")

    # Carrega o áudio
    audio, sr = sf.read(str(caminho_wav))

    # Se estéreo, mantém; se mono, pyloudnorm lida corretamente
    meter = pyln.Meter(sr)
    loudness_atual = meter.integrated_loudness(audio)

    # Se o áudio é silêncio puro (loudness -inf), não normaliza
    if loudness_atual == float("-inf"):
        logger.warning("Áudio com loudness -inf (possível silêncio), não normalizando")
        return

    # Normaliza para o alvo
    audio_normalizado = pyln.normalize.loudness(audio, loudness_atual, target_lufs)

    # Resalva no mesmo arquivo
    sf.write(str(caminho_wav), audio_normalizado, sr)

    logger.info(
        "Loudness normalizado de %.1f LUFS para %.1f LUFS -> %s",
        loudness_atual,
        target_lufs,
        caminho_wav,
    )
