import subprocess
import sys

TEXT = sys.argv[1] if len(sys.argv) > 1 else "Texto de teste para gerar o áudio."
REF_AUDIO = "/data/reference/minha_voz.wav"
F5_OUTPUT = "/data/f5_output/gerado.wav"
FINAL_OUTPUT = "/data/final_output/resultado.wav"

def run_in_container(container, cmd):
    full_cmd = ["docker", "exec", container, "bash", "-c", cmd]
    print(f"[{container}] {cmd}")
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Falha no container {container}")

def step_1_f5tts():
    cmd = (
        f'f5-tts_infer-cli '
        f'--ref_audio "{REF_AUDIO}" '
        f'--ref_text "" '  # opcional: transcrição do áudio de referência
        f'--gen_text "{TEXT}" '
        f'--output_dir "/data/f5_output"'
    )
    run_in_container("f5tts", cmd)

def step_2_seedvc():
    # seed-vc: converte o timbre do F5_OUTPUT para o timbre do REF_AUDIO
    cmd = (
        f'python3 inference.py '
        f'--source "{F5_OUTPUT}" '
        f'--target "{REF_AUDIO}" '
        f'--output "/data/final_output"'
    )
    run_in_container("seedvc", cmd)

if __name__ == "__main__":
    print("== Etapa 1: F5-TTS (prosódia) ==")
    step_1_f5tts()

    print("== Etapa 2: seed-vc (transferência de timbre) ==")
    step_2_seedvc()

    print(f"Pronto! Resultado em {FINAL_OUTPUT}")