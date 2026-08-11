import subprocess
import sys

TEXT = sys.argv[1] if len(sys.argv) > 1 else "Texto de teste para gerar o áudio."
RUN_NAME = sys.argv[2] if len(sys.argv) > 2 else "minha_voz_v1"

REF_AUDIO = "/data/reference/minha_voz.wav"  # ainda necessário mesmo com checkpoint fine-tunado
F5_OUTPUT = "/data/f5_output/gerado.wav"
FINAL_OUTPUT = "/data/final_output/resultado.wav"

# Checkpoint/config gerados pelo train_voice.py (copiados para /data/checkpoints)
CHECKPOINT = f"/data/checkpoints/ft_model.pth"
CONFIG = f"/data/checkpoints/{RUN_NAME}.yml"


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


def step_2_seedvc(use_finetuned=True):
    # seed-vc: converte o timbre do F5_OUTPUT para o timbre do REF_AUDIO,
    # usando o checkpoint fine-tunado (se existir) em vez do modelo genérico
    cmd = (
        f'python3 inference.py '
        f'--source "{F5_OUTPUT}" '
        f'--target "{REF_AUDIO}" '
        f'--output "/data/final_output" '
        f'--diffusion-steps 30 '
        f'--length-adjust 1.0 '
        f'--inference-cfg-rate 0.7 '
        f'--f0-condition False '
        f'--fp16 True'
    )
    if use_finetuned:
        cmd += f' --checkpoint "{CHECKPOINT}" --config "{CONFIG}"'
    run_in_container("seedvc", cmd)


if __name__ == "__main__":
    print("== Etapa 1: F5-TTS (prosódia) ==")
    step_1_f5tts()

    print(f"== Etapa 2: seed-vc fine-tunado ({RUN_NAME}) (transferência de timbre) ==")
    step_2_seedvc(use_finetuned=True)

    print(f"Pronto! Resultado em {FINAL_OUTPUT}")