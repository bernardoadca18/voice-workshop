from __future__ import annotations

import hashlib
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orchestrator"))
import audio_postprocess as ap  # noqa: E402


SAMPLE_RATE = 48_000
DURATION = 3.0


def sine(amplitude: float, frequency: float = 1_000.0, phase: float = 0.0) -> np.ndarray:
    samples = np.arange(round(SAMPLE_RATE * DURATION))
    return amplitude * np.sin(2.0 * np.pi * frequency * samples / SAMPLE_RATE + phase)


def peaky_signal() -> np.ndarray:
    signal = sine(0.018)
    signal[SAMPLE_RATE] = 0.82
    signal[2 * SAMPLE_RATE] = -0.82
    return signal


class AudioPostprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write(self, name: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE, subtype: str = "PCM_16") -> Path:
        path = self.root / name
        sf.write(path, audio, sample_rate, subtype=subtype)
        return path

    def test_reduces_gain_to_target_without_clipping(self) -> None:
        path = self.write("loud.wav", sine(0.7))
        result = ap.postprocess_audio(path)
        self.assertEqual(result.status, "PASS")
        self.assertLess(result.applied_gain_db, 0.0)
        self.assertAlmostEqual(result.after.integrated_lufs, -16.0, delta=0.08)
        self.assertLess(result.after.sample_peak_dbfs, 0.0)

    def test_raises_gain_to_target_when_peak_allows(self) -> None:
        path = self.write("quiet.wav", sine(0.09))
        result = ap.postprocess_audio(path)
        self.assertEqual(result.status, "PASS")
        self.assertGreater(result.applied_gain_db, 0.0)
        self.assertAlmostEqual(result.after.integrated_lufs, -16.0, delta=0.08)
        self.assertLessEqual(result.after.true_peak_dbtp, -0.98)

    def test_peak_limited_result_is_warning_and_not_error(self) -> None:
        path = self.write("peaky.wav", peaky_signal())
        result = ap.postprocess_audio(path)
        self.assertEqual(result.status, "WARNING")
        self.assertLess(result.applied_gain_db, result.gain_for_loudness_db)
        self.assertLess(result.after.integrated_lufs, -16.0)
        self.assertLessEqual(result.after.true_peak_dbtp, -0.98)
        self.assertLess(result.after.sample_peak_dbfs, 0.0)

    def test_output_is_reopened_and_reanalyzed(self) -> None:
        path = self.write("reanalyzed.wav", sine(0.12))
        original = ap._analyze_audio
        with mock.patch.object(ap, "_analyze_audio", wraps=original) as analyze:
            ap.postprocess_audio(path)
        self.assertEqual(analyze.call_count, 2)
        self.assertNotEqual(Path(analyze.call_args_list[0].args[0]), Path(analyze.call_args_list[1].args[0]))

    def test_true_peak_is_independent_and_exceeds_sample_peak(self) -> None:
        # A 12 kHz/48 kHz com fase pi/4, as amostras ficam em ±0,707 A,
        # enquanto a reconstrução entre amostras se aproxima de A.
        path = self.write("intersample.wav", sine(0.5, 12_000.0, math.pi / 4.0), subtype="FLOAT")
        metrics, _, _ = ap._analyze_audio(path)
        self.assertGreater(metrics.true_peak_dbtp, metrics.sample_peak_dbfs + 2.0)

    def test_true_peak_failure_is_explicit_and_original_is_untouched(self) -> None:
        path = self.write("failure.wav", sine(0.15))
        before = hashlib.sha256(path.read_bytes()).digest()
        with mock.patch.object(ap, "_measure_true_peak_db", side_effect=ap.AudioPostprocessError("TP failure")):
            with self.assertRaisesRegex(ap.AudioPostprocessError, "TP failure"):
                ap.postprocess_audio(path)
        self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), before)

    def test_sample_rate_channels_and_subtype_are_preserved(self) -> None:
        mono = sine(0.11)
        stereo = np.column_stack((mono, mono * 0.8))
        path = self.write("stereo.wav", stereo, sample_rate=SAMPLE_RATE, subtype="PCM_24")
        before = sf.info(path)
        ap.postprocess_audio(path)
        after = sf.info(path)
        self.assertEqual((after.samplerate, after.channels, after.subtype),
                         (before.samplerate, before.channels, before.subtype))

    def test_processing_is_deterministic(self) -> None:
        source = self.write("source.wav", sine(0.13))
        first = self.root / "first.wav"
        second = self.root / "second.wav"
        shutil.copyfile(source, first)
        shutil.copyfile(source, second)
        result_a = ap.postprocess_audio(first)
        result_b = ap.postprocess_audio(second)
        audio_a, _ = sf.read(first)
        audio_b, _ = sf.read(second)
        np.testing.assert_array_equal(audio_a, audio_b)
        self.assertEqual(result_a, result_b)

    def test_saved_output_contains_no_full_scale_samples(self) -> None:
        path = self.write("no_clip.wav", peaky_signal())
        result = ap.postprocess_audio(path)
        saved, _ = sf.read(path)
        self.assertLess(float(np.max(np.abs(saved))), 1.0)
        self.assertLess(result.after.sample_peak_dbfs, 0.0)

    def test_true_peak_overshoot_rejects_temp_and_preserves_original(self) -> None:
        path = self.write("unsafe.wav", sine(0.1))
        before = path.read_bytes()
        real_analyze = ap._analyze_audio
        calls = 0

        def unsafe_second_analysis(candidate: Path):
            nonlocal calls
            metrics, audio, info = real_analyze(candidate)
            calls += 1
            if calls == 2:
                metrics = ap.AudioMetrics(metrics.integrated_lufs, metrics.sample_peak_dbfs, -0.5)
            return metrics, audio, info

        with mock.patch.object(ap, "_analyze_audio", side_effect=unsafe_second_analysis):
            with self.assertRaisesRegex(ap.AudioPostprocessError, "True Peak excedeu"):
                ap.postprocess_audio(path)
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
