"""Pós-processamento final, transparente e peak-safe para narrações."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

logger = logging.getLogger("audio_postprocess")

TARGET_LUFS = float(os.environ.get("AUDIO_TARGET_LUFS", "-16.0"))
MAX_TRUE_PEAK_DB = float(os.environ.get("AUDIO_MAX_TRUE_PEAK_DB", "-1.0"))
TRUE_PEAK_TOLERANCE_DB = float(os.environ.get("AUDIO_TRUE_PEAK_TOLERANCE_DB", "0.02"))
TRUE_PEAK_SAFETY_MARGIN_DB = float(os.environ.get("AUDIO_TRUE_PEAK_SAFETY_MARGIN_DB", "0.05"))
MIN_ACCEPTABLE_LUFS = float(os.environ.get("AUDIO_MIN_ACCEPTABLE_LUFS", "-18.0"))
MAX_ACCEPTABLE_LUFS = float(os.environ.get("AUDIO_MAX_ACCEPTABLE_LUFS", "-15.0"))


class AudioPostprocessError(RuntimeError):
    """Falha que torna inseguro substituir o WAV de entrada."""


@dataclass(frozen=True)
class AudioMetrics:
    integrated_lufs: float
    sample_peak_dbfs: float
    true_peak_dbtp: float


@dataclass(frozen=True)
class PostprocessResult:
    before: AudioMetrics
    after: AudioMetrics
    target_lufs: float
    max_true_peak_db: float
    gain_for_loudness_db: float
    gain_allowed_by_peak_db: float
    applied_gain_db: float
    status: str
    reason: str


def _amplitude_to_db(amplitude: float) -> float:
    return 20.0 * math.log10(amplitude) if amplitude > 0.0 else float("-inf")


def _measure_true_peak_db(caminho_wav: Path) -> float:
    """Mede dBTP real com oversampling do filtro EBU R128 do FFmpeg.

    ``lavfi.r128.true_peak`` é uma amplitude linear, emitida a cada 100 ms
    com três casas decimais. Usamos o máximo e convertemos para dBTP. Isso
    oferece resolução próxima de 0,01 dB ao redor de 0 dBTP, bem superior ao
    summary textual (0,1 dB) do filtro.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioPostprocessError("FFmpeg não encontrado; True Peak não pôde ser medido")

    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(caminho_wav),
        "-map", "0:a:0",
        "-filter:a",
        "ebur128=metadata=1:peak=true,"
        "ametadata=mode=print:key=lavfi.r128.true_peak:file=-",
        "-f", "null",
        "-",
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise AudioPostprocessError(f"Falha ao executar FFmpeg para medir True Peak: {exc}") from exc

    if process.returncode != 0:
        detail = process.stderr.strip() or "sem detalhes"
        raise AudioPostprocessError(f"FFmpeg falhou ao medir True Peak: {detail}")

    prefix = "lavfi.r128.true_peak="
    try:
        amplitudes = [
            float(line.removeprefix(prefix))
            for line in process.stdout.splitlines()
            if line.startswith(prefix)
        ]
    except ValueError as exc:
        raise AudioPostprocessError("Saída inválida do FFmpeg ao medir True Peak") from exc

    if not amplitudes or not all(math.isfinite(value) and value >= 0.0 for value in amplitudes):
        raise AudioPostprocessError("FFmpeg não retornou uma medição válida de True Peak")
    return _amplitude_to_db(max(amplitudes))


def _analyze_audio(caminho_wav: Path) -> tuple[AudioMetrics, np.ndarray, sf.SoundFile]:
    try:
        info = sf.info(str(caminho_wav))
        audio, sample_rate = sf.read(str(caminho_wav), dtype="float64", always_2d=False)
    except (RuntimeError, OSError) as exc:
        raise AudioPostprocessError(f"WAV inválido ou ilegível: {caminho_wav}: {exc}") from exc

    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise AudioPostprocessError(f"WAV vazio ou com amostras inválidas: {caminho_wav}")

    try:
        loudness = float(pyln.Meter(sample_rate).integrated_loudness(audio))
    except (ValueError, RuntimeError) as exc:
        raise AudioPostprocessError(f"Falha ao medir Integrated LUFS: {exc}") from exc
    if not math.isfinite(loudness):
        raise AudioPostprocessError("Integrated LUFS inválido (áudio silencioso ou curto demais)")

    sample_peak = _amplitude_to_db(float(np.max(np.abs(audio))))
    true_peak = _measure_true_peak_db(caminho_wav)
    return AudioMetrics(loudness, sample_peak, true_peak), audio, info


def _classify_result(
    after: AudioMetrics,
    peak_constrained: bool,
    max_true_peak_db: float,
    true_peak_tolerance_db: float,
) -> tuple[str, str]:
    if after.sample_peak_dbfs >= 0.0:
        return "ERROR", "Clipping digital detectado no WAV salvo."
    if after.true_peak_dbtp > max_true_peak_db + true_peak_tolerance_db:
        return "ERROR", "True Peak excedeu o teto configurado além da tolerância."
    if after.integrated_lufs > MAX_ACCEPTABLE_LUFS:
        return "ERROR", "Loudness final acima da faixa aceitável."
    if peak_constrained:
        return "WARNING", "Target loudness não atingido porque o teto de True Peak teve prioridade."
    if after.integrated_lufs < MIN_ACCEPTABLE_LUFS:
        return "ERROR", "Loudness final abaixo da faixa aceitável sem limitação por True Peak."
    return "PASS", "Loudness e True Peak dentro dos limites."


def postprocess_audio(
    caminho_wav: str | Path,
    target_lufs: float = TARGET_LUFS,
    max_true_peak_db: float = MAX_TRUE_PEAK_DB,
    true_peak_tolerance_db: float = TRUE_PEAK_TOLERANCE_DB,
    true_peak_safety_margin_db: float = TRUE_PEAK_SAFETY_MARGIN_DB,
) -> PostprocessResult:
    """Aplica somente ganho constante, valida o WAV salvo e o substitui atomicamente."""
    caminho_wav = Path(caminho_wav)
    if not caminho_wav.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_wav}")
    if true_peak_tolerance_db < 0.0 or true_peak_safety_margin_db < 0.0:
        raise ValueError("Tolerância e safety margin de True Peak não podem ser negativas")

    before, audio, info = _analyze_audio(caminho_wav)
    gain_for_loudness = target_lufs - before.integrated_lufs
    processing_ceiling = max_true_peak_db - true_peak_safety_margin_db
    gain_allowed_by_peak = processing_ceiling - before.true_peak_dbtp
    applied_gain = min(gain_for_loudness, gain_allowed_by_peak)
    peak_constrained = gain_allowed_by_peak < gain_for_loudness - 1e-9
    processed = audio * (10.0 ** (applied_gain / 20.0))

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{caminho_wav.stem}.", suffix=".wav", dir=caminho_wav.parent, delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
        sf.write(
            str(temp_path),
            processed,
            info.samplerate,
            format=info.format,
            subtype=info.subtype,
            endian=info.endian,
        )

        # Mede o arquivo codificado, fechado e reaberto; não métricas projetadas.
        after, _, after_info = _analyze_audio(temp_path)
        if after_info.samplerate != info.samplerate or after_info.channels != info.channels:
            raise AudioPostprocessError("Sample rate ou número de canais mudou durante o pós-processamento")
        if after_info.subtype != info.subtype:
            raise AudioPostprocessError("Subtype/bit depth mudou durante o pós-processamento")

        status, reason = _classify_result(
            after, peak_constrained, max_true_peak_db, true_peak_tolerance_db
        )
        result = PostprocessResult(
            before=before,
            after=after,
            target_lufs=target_lufs,
            max_true_peak_db=max_true_peak_db,
            gain_for_loudness_db=gain_for_loudness,
            gain_allowed_by_peak_db=gain_allowed_by_peak,
            applied_gain_db=applied_gain,
            status=status,
            reason=reason,
        )
        _log_result(caminho_wav, result, processing_ceiling, true_peak_tolerance_db)
        if status == "ERROR":
            raise AudioPostprocessError(reason)

        os.replace(temp_path, caminho_wav)
        temp_path = None
        return result
    except AudioPostprocessError:
        raise
    except (OSError, RuntimeError) as exc:
        raise AudioPostprocessError(f"Falha ao salvar o WAV pós-processado: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _log_result(path: Path, result: PostprocessResult, processing_ceiling: float, tolerance: float) -> None:
    logger.info(
        "Audio post-processing: %s\n"
        "Before: LUFS %.2f | Sample Peak %.2f dBFS | True Peak %.2f dBTP\n"
        "Target %.2f LUFS | Max True Peak %.2f dBTP | Processing ceiling %.2f dBTP | Tolerance %.2f dB\n"
        "Gain: loudness %+.2f dB | peak allowed %+.2f dB | applied %+.2f dB\n"
        "After: LUFS %.2f | Sample Peak %.2f dBFS | True Peak %.2f dBTP\n"
        "Status: %s | Reason: %s",
        path,
        result.before.integrated_lufs,
        result.before.sample_peak_dbfs,
        result.before.true_peak_dbtp,
        result.target_lufs,
        result.max_true_peak_db,
        processing_ceiling,
        tolerance,
        result.gain_for_loudness_db,
        result.gain_allowed_by_peak_db,
        result.applied_gain_db,
        result.after.integrated_lufs,
        result.after.sample_peak_dbfs,
        result.after.true_peak_dbtp,
        result.status,
        result.reason,
    )


def normalizar_loudness(
    caminho_wav: str | Path, target_lufs: float = TARGET_LUFS
) -> PostprocessResult:
    """Compatibilidade com o call site histórico do orchestrator."""
    return postprocess_audio(caminho_wav, target_lufs=target_lufs)
