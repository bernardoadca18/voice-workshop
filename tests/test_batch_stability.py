from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "seedvc" / "server.py"


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def empty_cache() -> None:
        return None


def load_server(fake_loader):
    torch = ModuleType("torch")
    torch.cuda = FakeCuda()
    torch.set_num_threads = lambda _: None
    torch.set_num_interop_threads = lambda _: None
    torch.inference_mode = lambda: (lambda function: function)

    inference = ModuleType("inference")
    inference.load_models = fake_loader
    inference.crossfade = lambda *args: args[1]
    inference.adjust_f0_semitones = lambda value, _: value
    inference.device = SimpleNamespace(type="cuda")

    fake_modules = {
        "torch": torch,
        "torchaudio": ModuleType("torchaudio"),
        "librosa": ModuleType("librosa"),
        "inference": inference,
    }
    old_modules = {name: sys.modules.get(name) for name in fake_modules}
    sys.modules.update(fake_modules)
    old_limit = os.environ.get("SEEDVC_MAX_CACHED_MODELS")
    os.environ["SEEDVC_MAX_CACHED_MODELS"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(
            f"seedvc_server_test_{time.monotonic_ns()}", SERVER_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        module._cleanup_runtime = lambda _: None
        return module
    finally:
        if old_limit is None:
            os.environ.pop("SEEDVC_MAX_CACHED_MODELS", None)
        else:
            os.environ["SEEDVC_MAX_CACHED_MODELS"] = old_limit
        for name, previous in old_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class SeedVCBatchStabilityTests(unittest.TestCase):
    def test_concurrent_requests_load_the_same_model_once(self) -> None:
        calls = 0
        calls_lock = threading.Lock()
        bundle = tuple(object() for _ in range(7))

        def fake_loader(_):
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.05)
            return bundle

        server = load_server(fake_loader)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(lambda _: server._carregar("zero_shot", True), range(4))
            )

        self.assertEqual(calls, 1)
        self.assertTrue(all(result is bundle for result in results))
        self.assertEqual(list(server._modelo_cache), [("zero_shot", True)])

    def test_switching_precision_evicts_the_previous_model(self) -> None:
        loads = []

        def fake_loader(args):
            bundle = tuple(object() for _ in range(7))
            loads.append((args.fp16, bundle))
            return bundle

        server = load_server(fake_loader)
        first = server._carregar("zero_shot", True)
        server._ref_cache[(("zero_shot", True), "/voice.wav", 1, 1)] = ("cached",)
        second = server._carregar("zero_shot", False)

        self.assertIsNot(first, second)
        self.assertEqual(len(loads), 2)
        self.assertEqual(list(server._modelo_cache), [("zero_shot", False)])
        self.assertFalse(server._ref_cache)

    def test_convert_endpoint_serializes_concurrent_inference(self) -> None:
        server = load_server(lambda _: tuple(object() for _ in range(7)))
        active = 0
        maximum_active = 0
        state_lock = threading.Lock()

        def fake_converter(*args, **kwargs):
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1
            return args[2]

        server.converter = fake_converter
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            target = Path(directory) / "target.wav"
            source.touch()
            target.touch()

            def request(index):
                return server.ConvertRequest(
                    source_path=str(source),
                    target_path=str(target),
                    output_path=str(Path(directory) / f"out-{index}.wav"),
                    run_name="zero_shot",
                    diffusion_steps=1,
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(
                    lambda index: server.convert_endpoint(request(index)),
                    range(2),
                ))

        self.assertEqual(maximum_active, 1)
        self.assertEqual(len(results), 2)

    def test_cleanup_failure_still_releases_inference_lock(self) -> None:
        server = load_server(lambda _: tuple(object() for _ in range(7)))
        server.converter = mock.Mock(return_value="/tmp/out.wav")
        server._cleanup_runtime = mock.Mock(side_effect=RuntimeError("cleanup failed"))

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            target = Path(directory) / "target.wav"
            source.touch()
            target.touch()
            request = server.ConvertRequest(
                source_path=str(source),
                target_path=str(target),
                output_path=str(Path(directory) / "out.wav"),
                run_name="zero_shot",
            )
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                server.convert_endpoint(request)

        self.assertTrue(server._resource_lock.acquire(blocking=False))
        server._resource_lock.release()

    def test_terminal_job_history_is_pruned_for_long_batches(self) -> None:
        jobs_path = ROOT / "orchestrator" / "jobs.py"
        spec = importlib.util.spec_from_file_location(
            f"jobs_test_{time.monotonic_ns()}", jobs_path
        )
        jobs = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(jobs)
        jobs.MAX_RETAINED_JOBS = 2

        first = jobs.criar_job()
        jobs.atualizar_job(first.id, status=jobs.JobStatus.CONCLUIDO)
        second = jobs.criar_job()
        jobs.atualizar_job(second.id, status=jobs.JobStatus.ERRO)
        third = jobs.criar_job()

        self.assertIsNone(jobs.obter_job(first.id))
        self.assertIsNotNone(jobs.obter_job(second.id))
        self.assertIsNotNone(jobs.obter_job(third.id))
        self.assertEqual(jobs.quantidade_jobs(), 2)

    def test_seedvc_read_timeout_is_not_retried(self) -> None:
        orchestrator_path = str(ROOT / "orchestrator")
        sys.path.insert(0, orchestrator_path)
        stubs = {
            "audio_chunker": ModuleType("audio_chunker"),
            "audio_postprocess": ModuleType("audio_postprocess"),
            "glossario_fonetico": ModuleType("glossario_fonetico"),
        }
        stubs["audio_chunker"].fatiar_por_silencio = mock.Mock()
        stubs["audio_chunker"].salvar_chunks = mock.Mock()
        stubs["audio_chunker"].concatenar_com_crossfade = mock.Mock()
        stubs["audio_postprocess"].normalizar_loudness = mock.Mock()
        stubs["glossario_fonetico"].aplicar_respelling = lambda text: text
        previous = {name: sys.modules.get(name) for name in stubs}
        sys.modules.update(stubs)
        sys.modules.pop("run_pipeline", None)
        try:
            import run_pipeline
            import requests

            converted_dir = Path("/tmp/nonexistent-converted-dir")
            with mock.patch.object(
                run_pipeline._http_session,
                "post",
                side_effect=requests.ReadTimeout("still running remotely"),
            ) as post:
                with self.assertRaises(requests.ReadTimeout):
                    run_pipeline.converter_chunk_seedvc(
                        "/data/source.wav",
                        "zero_shot",
                        1,
                        converted_dir,
                        "/data/output",
                        1,
                    )
            self.assertEqual(post.call_count, 1)
        finally:
            sys.modules.pop("run_pipeline", None)
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
            sys.path.remove(orchestrator_path)


if __name__ == "__main__":
    unittest.main()
