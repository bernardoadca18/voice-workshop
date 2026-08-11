import subprocess
import sys
import os
import requests
import glob

TEXT = sys.argv[1] if len(sys.argv) > 1 else "Texto de teste para gerar o áudio."
RUN_NAME = sys.argv[2] if len(sys.argv) > 2 else "minha_voz_v1"

KOKORO_URL = "http://localhost:8880/v1/audio/speech"
KOKORO_API_KEY = os.environ.get("KOKORO_API_KEY", "")
KOKORO_VOICE = "pm_alex"

REF_AUDIO_MINHA_VOZ = "/data/reference/voz_referencia.wav"
F5_OUTPUT_HOST = "./data/f5_output/gerado.wav"
F5_OUTPUT_CONTAINER = "/data/f5_output/gerado.wav"
FINAL_OUTPUT = "/data/final_output/resultado.wav"

CHECKPOINT = "/data/checkpoints/ft_model.pth"
CONFIG = f"/data/checkpoints/{RUN_NAME}.yml"


def run_in_container(container, cmd):
    full_cmd = ["docker", "exec", container, "bash", "-c", cmd]
    print(f"[{container}] {cmd}")
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Falha no container {container}")


def step_1_kokoro():
    headers = {"Content-Type": "application/json"}
    if KOKORO_API_KEY:
        headers["Authorization"] = f"Bearer {KOKORO_API_KEY}"

    resp = requests.post(
        KOKORO_URL,
        headers=headers,
        json={
            "model": "tts-1",
            "input": TEXT,
            "voice": KOKORO_VOICE,
            "response_format": "wav",
        },
        timeout=60,
    )
    resp.raise_for_status()

    os.makedirs(os.path.dirname(F5_OUTPUT_HOST), exist_ok=True)
    with open(F5_OUTPUT_HOST, "wb") as f:
        f.write(resp.content)
    print(f"[kokoro] áudio base salvo em {F5_OUTPUT_HOST} ({len(resp.content)} bytes)")


def step_2_seedvc(use_finetuned=True):
    cmd = (
        f'python3 inference.py '
        f'--source "{F5_OUTPUT_CONTAINER}" '
        f'--target "{REF_AUDIO_MINHA_VOZ}" '
        f'--output "/data/final_output" '
        f'--diffusion-steps 50 '
        f'--length-adjust 1.0 '
        f'--inference-cfg-rate 0.7 '
        f'--f0-condition False '
        f'--fp16 True'
    )
    if use_finetuned:
        cmd += f' --checkpoint "{CHECKPOINT}" --config "{CONFIG}"'
    run_in_container("seedvc", cmd)

def obter_ultimo_output():
    arquivos = glob.glob("./data/final_output/vc_*.wav")
    if not arquivos:
        raise RuntimeError("Nenhum arquivo de saída encontrado em ./data/final_output/")
    return max(arquivos, key=lambda p: __import__("os").path.getmtime(p))

if __name__ == "__main__":
    if not KOKORO_API_KEY:
        print("AVISO: KOKORO_API_KEY não definida no ambiente — a chamada vai falhar com 401 se a API exigir auth.", file=sys.stderr)

    print("== Etapa 1: Kokoro (prosódia, via API HTTP) ==")
    step_1_kokoro()

    print(f"== Etapa 2: seed-vc fine-tunado ({RUN_NAME}) (transferência de timbre) ==")
    step_2_seedvc(use_finetuned=True)

    resultado = obter_ultimo_output()
    print(f"Pronto! Resultado em {resultado}")