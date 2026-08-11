"""
Fine-tuning do seed-vc na própria voz.

Uso:
    python3 train_voice.py [run_name] [max_steps]

Pré-requisito: os 64 (ou quantos forem) arquivos .wav tratados devem estar em
    data/train_dataset/
Não precisa de subpastas nem transcrição -- só os áudios de 1-30s cada.
"""
import subprocess
import sys

RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "minha_voz_v1"
MAX_STEPS = sys.argv[2] if len(sys.argv) > 2 else "1000"

# Config de fine-tuning para voice conversion "offline" (não singing, não real-time)
CONFIG = "./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
DATASET_DIR = "/data/train_dataset"


def run_in_container(container, cmd):
    full_cmd = ["docker", "exec", "-it", container, "bash", "-c", cmd]
    print(f"[{container}] {cmd}")
    result = subprocess.run(full_cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Falha no container {container}")


def train():
    cmd = (
        f'python3 train.py '
        f'--config "{CONFIG}" '
        f'--dataset-dir "{DATASET_DIR}" '
        f'--run-name "{RUN_NAME}" '
        f'--batch-size 2 '
        f'--max-steps {MAX_STEPS} '
        f'--max-epochs 1000 '
        f'--save-every 100 '
        f'--num-workers 0'
    )
    run_in_container("seedvc", cmd)


if __name__ == "__main__":
    print(f"== Fine-tuning seed-vc | run-name={RUN_NAME} | max-steps={MAX_STEPS} ==")
    train()
    print(
        f"Concluído. Checkpoint deve estar em "
        f"./runs/{RUN_NAME}/ft_model.pth dentro do container seedvc "
        f"(equivalente a /app/runs/{RUN_NAME}/ft_model.pth)."
    )
    print(
        "Copie o checkpoint e o config para /data/checkpoints/ para reuso fora do container, ex:\n"
        f"  docker exec seedvc cp /app/runs/{RUN_NAME}/ft_model.pth /data/checkpoints/\n"
        f"  docker exec seedvc cp {CONFIG} /data/checkpoints/{RUN_NAME}.yml"
    )