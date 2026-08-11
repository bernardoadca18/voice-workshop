"""
Fatiamento de áudio por silêncio natural — usado para dividir a saída
contínua do Kokoro em pedaços de 4-8s antes do Seed-VC, sem nunca
fatiar o TEXTO (isso preservaria o problema de fragmentação de prosódia
que o pipeline como um todo foi desenhado para evitar).

Estratégia: detecta silêncios reais na fala já sintetizada e agrupa
segmentos entre eles até atingir a janela alvo de duração, cortando
"a força" apenas se um trecho contínuo sem pausas ultrapassar o máximo
permitido (caso raro, mas tratado para não travar o pipeline).
"""
import logging
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_silence

logger = logging.getLogger("audio_chunker")

MIN_CHUNK_MS = 4_000
MAX_CHUNK_MS = 8_000
MIN_SILENCE_LEN_MS = 300
SILENCE_THRESH_OFFSET_DB = 16  # abaixo da média (dBFS) do próprio áudio
DESCARTE_MINIMO_MS = 500  # sobras menores que isso no fim são descartadas


def fatiar_por_silencio(
    caminho_audio: str | Path,
    min_chunk_ms: int = MIN_CHUNK_MS,
    max_chunk_ms: int = MAX_CHUNK_MS,
    min_silence_len_ms: int = MIN_SILENCE_LEN_MS,
) -> list[AudioSegment]:
    """
    Divide um áudio contínuo em pedaços entre min_chunk_ms e max_chunk_ms,
    cortando preferencialmente em pontos de silêncio real da fala.

    Returns:
        Lista de AudioSegment, na ordem original.
    """
    audio = AudioSegment.from_file(caminho_audio)
    duracao_total = len(audio)

    # Threshold adaptativo: relativo ao volume médio do próprio áudio,
    # em vez de um dBFS fixo (que variaria demais entre vozes/gravações).
    silence_thresh = audio.dBFS - SILENCE_THRESH_OFFSET_DB
    silencios = detect_silence(
        audio, min_silence_len=min_silence_len_ms, silence_thresh=silence_thresh
    )
    pontos_candidatos = [(s + e) // 2 for s, e in silencios]

    cortes: list[int] = []
    inicio_chunk = 0

    for ponto in pontos_candidatos:
        duracao_atual = ponto - inicio_chunk

        if duracao_atual < min_chunk_ms:
            # silêncio cedo demais -- ignora e continua acumulando
            continue

        if duracao_atual <= max_chunk_ms:
            cortes.append(ponto)
            inicio_chunk = ponto
            continue

        # Passou do máximo sem ponto de silêncio adequado no meio --
        # corte forçado, sem pausa natural. Log de aviso porque isso
        # tende a soar pior (corte no meio da fala), mas é preferível
        # a mandar um bloco gigante pro Seed-VC.
        corte_forcado = inicio_chunk + max_chunk_ms
        logger.warning(
            "audio_chunker.corte_forcado",
            extra={"em_ms": corte_forcado, "motivo": "sem silêncio no intervalo permitido"},
        )
        cortes.append(corte_forcado)
        inicio_chunk = corte_forcado

        # reavalia o mesmo ponto de silêncio contra o novo início
        duracao_atual = ponto - inicio_chunk
        if min_chunk_ms <= duracao_atual <= max_chunk_ms:
            cortes.append(ponto)
            inicio_chunk = ponto

    limites = [0] + cortes + [duracao_total]
    chunks = []
    for i in range(len(limites) - 1):
        ini, fim = limites[i], limites[i + 1]
        if fim - ini < DESCARTE_MINIMO_MS:
            continue
        chunks.append(audio[ini:fim])

    logger.info(
        "audio_chunker.concluido",
        extra={"total_chunks": len(chunks), "duracao_total_ms": duracao_total},
    )
    return chunks


def salvar_chunks(chunks: list[AudioSegment], diretorio_saida: str | Path) -> list[Path]:
    """Salva os chunks em disco, numerados sequencialmente. Limpa o diretório antes."""
    diretorio_saida = Path(diretorio_saida)
    diretorio_saida.mkdir(parents=True, exist_ok=True)

    for antigo in diretorio_saida.glob("chunk_*.wav"):
        antigo.unlink()

    paths = []
    for i, chunk in enumerate(chunks, start=1):
        path = diretorio_saida / f"chunk_{i:03d}.wav"
        chunk.export(path, format="wav")
        paths.append(path)

    return paths


def concatenar_com_crossfade(
    caminhos_convertidos: list[str | Path],
    caminho_saida: str | Path,
    crossfade_ms: int = 80,
) -> Path:
    """
    Concatena os pedaços já convertidos pelo Seed-VC, aplicando um
    crossfade curto entre eles em vez de emenda seca -- suaviza qualquer
    descontinuidade de energia/timbre entre conversões feitas de forma
    independente por chunk.
    """
    if not caminhos_convertidos:
        raise ValueError("Nenhum arquivo convertido para concatenar")

    resultado = AudioSegment.from_file(caminhos_convertidos[0])
    for caminho in caminhos_convertidos[1:]:
        proximo = AudioSegment.from_file(caminho)
        # crossfade não pode ser maior que o menor dos dois segmentos
        cf = min(crossfade_ms, len(resultado), len(proximo))
        resultado = resultado.append(proximo, crossfade=cf)

    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    resultado.export(caminho_saida, format="wav")
    return caminho_saida