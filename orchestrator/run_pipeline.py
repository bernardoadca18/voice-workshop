import os
import requests

SEEDVC_URL = os.environ.get("SEEDVC_URL", "http://localhost:8001")

def converter_chunk_seedvc(
    chunk_path_container: str,
    run_name: str,
    indice: int,
    converted_dir_host: Path,
    converted_dir_container: str,
) -> Path:
    saida_container = f"{converted_dir_container}/part_{indice:03d}.wav"
    resp = requests.post(
        f"{SEEDVC_URL}/convert",
        json={
            "source_path": chunk_path_container,
            "target_path": REF_AUDIO_MINHA_VOZ,
            "output_path": saida_container,
            "run_name": run_name,
            "diffusion_steps": DIFFUSION_STEPS,
        },
        timeout=SEEDVC_TIMEOUT_S,
    )
    resp.raise_for_status()
    return converted_dir_host / f"part_{indice:03d}.wav"