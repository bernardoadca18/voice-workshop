"""
Fine-tuning do seed-vc na própria voz.

Uso:
    python3 train_voice.py --dataset gold_v1 --run-name bernardo_gold_v1
    python3 train_voice.py --dataset full_v1 --run-name bernardo_full_v1

Pré-requisito: arquivos .wav tratados devem estar em
    data/train_datasets/<dataset_name>/

Não precisa de subpastas nem transcrição -- só os áudios de 1-30s cada.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Default training profile for RTX 2060 12GB VRAM
# Priority: stability > avoid CUDA OOM > reproducibility > speed
DEFAULT_TRAINING_PROFILE = {
    "batch_size": 2,
    "max_steps": 1000,
    "max_epochs": 1000,
    "save_every": 100,
    "num_workers": 0,
    "gpu": 0,
}

# Config de fine-tuning para voice conversion "offline" (não singing, não real-time)
CONFIG = "./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"

# Dataset root dentro do container
DATASETS_ROOT = "/data/train_datasets"

# Runs directory dentro do container
RUNS_DIR_CONTAINER = "/app/runs"

# Checkpoints directory dentro do container
CHECKPOINTS_DIR_CONTAINER = "/data/checkpoints"


def validate_run_name(run_name: str) -> None:
    """
    Valida o formato do run_name para evitar path traversal.
    Aceita apenas: letras, números, ponto, underscore, hífen.
    """
    if not re.match(r'^[a-zA-Z0-9._-]+$', run_name):
        raise ValueError(
            f"run_name inválido: '{run_name}'\n"
            "Use apenas letras, números, ponto (.), underscore (_) e hífen (-).\n"
            "Exemplos válidos: bernardo_gold_v1, bernardo-full-v1, voice.01"
        )


def validate_dataset_path(dataset_name: str) -> str:
    """
    Valida e retorna o path do dataset dentro do container.

    Args:
        dataset_name: Nome do dataset (ex: gold_v1, full_v1)

    Returns:
        Path absoluto do dataset dentro do container

    Raises:
        ValueError: Se dataset for inválido ou vazio
    """
    dataset_path = f"{DATASETS_ROOT}/{dataset_name}"

    # Proteção contra tentativa de usar o root dos datasets
    if dataset_name in ["", ".", "..", "train_datasets"]:
        raise ValueError(
            "Dataset path appears to be a dataset root.\n"
            "Select one specific dataset, e.g. gold_v1 or full_v1."
        )

    # Proteção contra path traversal
    if ".." in dataset_name or "/" in dataset_name or "\\" in dataset_name:
        raise ValueError(
            f"Dataset name contains invalid characters: '{dataset_name}'\n"
            "Use apenas o nome do dataset, ex: gold_v1"
        )

    return dataset_path


def check_container_running(container: str) -> bool:
    """Verifica se o container está rodando."""
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name={container}", "--filter", "status=running", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )
    return container in result.stdout


def check_gpu_preflight(container: str) -> None:
    """
    Executa verificação de GPU antes do treinamento.
    Falha se CUDA não estiver disponível (blocking error).
    """
    print("\nGPU Preflight Check")
    print("-" * 40)

    # Verifica se CUDA está disponível
    cmd = 'python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"'

    result = subprocess.run(
        ["docker", "exec", container, "bash", "-c", cmd],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: CUDA is NOT available!")
        print("Training requires CUDA-capable GPU with proper driver.")
        print("\nTroubleshooting:")
        print("  1. Check if GPU is visible: nvidia-smi")
        print("  2. Check Docker GPU access: docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi")
        print("  3. Ensure nvidia-container-toolkit is installed")
        raise RuntimeError("CUDA not available - cannot proceed with training")

    # CUDA está disponível - mostra detalhes
    cmd = 'python3 -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\'); print(f\'Device count: {torch.cuda.device_count()}\'); print(f\'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'N/A\'}\')"'

    result = subprocess.run(
        ["docker", "exec", container, "bash", "-c", cmd],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(result.stdout.strip())

    # Tenta obter info de VRAM via nvidia-smi
    vram_cmd = 'nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits'
    result = subprocess.run(
        ["docker", "exec", container, "bash", "-c", vram_cmd],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    name, total, free = parts[0], parts[1], parts[2]
                    print(f"GPU: {name}")
                    print(f"Total VRAM: {total} MB")
                    print(f"Free VRAM: {free} MB")
    else:
        print("Could not query GPU memory via nvidia-smi")

    print("-" * 40)


def check_kokoro_running() -> None:
    """Verifica se o Kokoro está rodando e emite warning se estiver."""
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=kokoro", "--filter", "status=running", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )

    if "kokoro" in result.stdout:
        print("\n" + "=" * 60)
        print("WARNING: Kokoro is currently running and may be occupying GPU memory.")
        print("For maximum training stability, stop GPU-consuming services before training.")
        print("\nSuggested:")
        print("  docker compose stop kokoro orchestrator")
        print("=" * 60 + "\n")


def count_audio_files(container: str, dataset_path: str) -> int:
    """Conta arquivos de áudio no dataset."""
    cmd = f'find "{dataset_path}" -maxdepth 1 -type f \\( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \\) | wc -l'
    result = subprocess.run(
        ["docker", "exec", container, "bash", "-c", cmd],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return 0

    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def check_final_checkpoint_exists(container: str, run_name: str) -> bool:
    """Verifica se já existe um checkpoint final para este run_name."""
    checkpoint_path = f"{CHECKPOINTS_DIR_CONTAINER}/{run_name}.pth"
    cmd = f'test -f "{checkpoint_path}" && echo "EXISTS" || echo "NOT_EXISTS"'
    result = subprocess.run(
        ["docker", "exec", container, "bash", "-c", cmd],
        capture_output=True,
        text=True
    )
    return "EXISTS" in result.stdout


def check_run_final_model_exists(container: str, run_name: str) -> bool:
    """Verifica se já existe ft_model.pth na run (treinamento finalizado)."""
    run_path = f"{RUNS_DIR_CONTAINER}/{run_name}/ft_model.pth"
    cmd = f'test -f "{run_path}" && echo "EXISTS" || echo "NOT_EXISTS"'
    result = subprocess.run(
        ["docker", "exec", container, "bash", "-c", cmd],
        capture_output=True,
        text=True
    )
    return "EXISTS" in result.stdout


def run_in_container(container: str, cmd: str, batch_size: int = None, check_oom: bool = True) -> None:
    """
    Executa comando no container EXATAMENTE UMA VEZ com streaming de output.
    Trata erros, incluindo CUDA OOM.

    Args:
        container: Nome do container
        cmd: Comando a executar
        batch_size: Batch size atual (para mensagem de OOM)
        check_oom: Se deve verificar CUDA OOM na saída

    CRITICAL: O comando executa EXATAMENTE UMA VEZ.
    Usa subprocess.Popen() para streaming em tempo real.
    """
    print(f"\n[{container}] {cmd}\n")

    # Usa Popen para streaming em tempo real - EXECUTA UMA ÚNICA VEZ
    process = subprocess.Popen(
        ["docker", "exec", "-i", container, "bash", "-c", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # Captura output em tempo real
    output_lines = []
    for line in process.stdout:
        print(line, end='')  # Stream para console
        output_lines.append(line)  # Salva para análise de OOM

    process.wait()
    output = ''.join(output_lines)

    if process.returncode != 0:
        # Verifica se foi CUDA OOM
        if check_oom:
            oom_patterns = [
                "CUDA out of memory",
                "OutOfMemoryError",
                "out of memory",
                "CUDA error: out of memory",
            ]

            for pattern in oom_patterns:
                if pattern.lower() in output.lower():
                    # Usa batch_size do argumento ou fallback para default
                    actual_batch_size = batch_size if batch_size is not None else DEFAULT_TRAINING_PROFILE['batch_size']

                    print("\n" + "=" * 60)
                    print("CUDA OUT OF MEMORY DETECTED")
                    print("=" * 60)
                    print(f"\nCurrent batch size: {actual_batch_size}")
                    print("\nRecommended retry:")
                    print("  --batch-size 1")
                    print("\nDo not change other training parameters when comparing Gold vs Full.")
                    print("=" * 60 + "\n")
                    raise RuntimeError("CUDA out of memory")

        raise RuntimeError(f"Falha no container {container} (exit code: {process.returncode})")


def reset_run(container: str, run_name: str) -> None:
    """
    Deleta completamente o diretório da run para treinamento do zero.
    Usado quando --reset-run é especificado.
    """
    run_dir = f"{RUNS_DIR_CONTAINER}/{run_name}"

    # Verifica se a run existe
    cmd = f'test -d "{run_dir}" && echo "EXISTS" || echo "NOT_EXISTS"'
    result = subprocess.run(
        ["docker", "exec", container, "bash", "-c", cmd],
        capture_output=True,
        text=True
    )

    if "EXISTS" in result.stdout:
        print(f"\nResetting run: {run_dir}")
        run_in_container(
            container,
            f'rm -rf "{run_dir}"',
            check_oom=False
        )
        print(f"✓ Run directory deleted")


def export_checkpoint(container: str, run_name: str, force: bool = False) -> None:
    """
    Exporta o checkpoint treinado para /data/checkpoints/.

    Args:
        container: Nome do container
        run_name: Nome da run
        force: Se deve sobrescrever checkpoint existente
    """
    run_dir = f"{RUNS_DIR_CONTAINER}/{run_name}"
    checkpoint_src = f"{run_dir}/ft_model.pth"
    checkpoint_dst = f"{CHECKPOINTS_DIR_CONTAINER}/{run_name}.pth"
    config_dst = f"{CHECKPOINTS_DIR_CONTAINER}/{run_name}.yml"

    # Verifica se checkpoint final existe
    cmd = f'test -f "{checkpoint_src}" && echo "EXISTS" || echo "NOT_EXISTS"'
    result = subprocess.run(
        ["docker", "exec", container, "bash", "-c", cmd],
        capture_output=True,
        text=True
    )

    if "NOT_EXISTS" in result.stdout:
        print(f"\nWarning: Final checkpoint not found: {checkpoint_src}")
        print("Training may not have completed successfully.")
        return

    # Verifica se destino já existe
    if not force:
        cmd = f'test -f "{checkpoint_dst}" && echo "EXISTS" || echo "NOT_EXISTS"'
        result = subprocess.run(
            ["docker", "exec", container, "bash", "-c", cmd],
            capture_output=True,
            text=True
        )

        if "EXISTS" in result.stdout:
            print(f"\nError: Checkpoint already exists: {checkpoint_dst}")
            print("Use --force to overwrite.")
            raise RuntimeError("Checkpoint exists, use --force to overwrite")

    # Copia checkpoint
    print(f"\nExporting checkpoint...")
    run_in_container(
        container,
        f'cp "{checkpoint_src}" "{checkpoint_dst}"',
        check_oom=False
    )

    # Copia config
    run_in_container(
        container,
        f'cp "{CONFIG}" "{config_dst}"',
        check_oom=False
    )

    # Verifica se arquivos foram criados
    print(f"\nVerifying export...")
    for path in [checkpoint_dst, config_dst]:
        cmd = f'test -f "{path}" && echo "EXISTS" || echo "NOT_EXISTS"'
        result = subprocess.run(
            ["docker", "exec", container, "bash", "-c", cmd],
            capture_output=True,
            text=True
        )
        if "NOT_EXISTS" in result.stdout:
            raise RuntimeError(f"Failed to create: {path}")

    print(f"✓ Checkpoint exported: {checkpoint_dst}")
    print(f"✓ Config exported: {config_dst}")


def train(args: argparse.Namespace) -> None:
    """Executa o treinamento com os parâmetros fornecidos."""
    container = "seedvc"

    # Verifica se container está rodando
    if not check_container_running(container):
        print(f"Error: Container '{container}' is not running.")
        print("Start it with: docker compose up -d seedvc")
        sys.exit(1)

    # Valida run_name
    try:
        validate_run_name(args.run_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Valida dataset
    try:
        dataset_path = validate_dataset_path(args.dataset)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Preflight checks
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"\nRun:             {args.run_name}")
    print(f"Dataset:         {dataset_path}")

    # Conta arquivos de áudio
    audio_count = count_audio_files(container, dataset_path)
    print(f"Audio files:     {audio_count}")

    if audio_count == 0:
        print(f"\nError: No audio files found in dataset: {dataset_path}")
        print("Dataset must contain WAV/MP3/FLAC files.")
        sys.exit(1)

    print(f"\nPreset:\n{Path(CONFIG).name}")
    print(f"\nBatch size:      {args.batch_size}")
    print(f"Max steps:       {args.max_steps}")
    print(f"Max epochs:      {args.max_epochs}")
    print(f"Save every:      {args.save_every}")
    print(f"Workers:         {args.num_workers}")
    print(f"GPU:             {args.gpu}")

    # Verifica se checkpoint final já existe
    if check_final_checkpoint_exists(container, args.run_name):
        if args.force:
            print(f"\nWarning: Will overwrite existing checkpoint: {args.run_name}.pth")
        else:
            print(f"\nError: Checkpoint already exists: /data/checkpoints/{args.run_name}.pth")
            print("Use --force to overwrite.")
            sys.exit(1)

    # Verifica se treinamento já foi finalizado
    if check_run_final_model_exists(container, args.run_name):
        if args.reset_run:
            print(f"\nNote: Run has existing model, but --reset-run will clear run history")
        elif args.force:
            print(f"\nWarning: Run already has final model.")
            print("Use --reset-run to clear run history and train from scratch.")
            print("Use --force only to overwrite the final checkpoint after training.")
        else:
            print(f"\nError: Run already finalized: /app/runs/{args.run_name}/ft_model.pth")
            print("Use --reset-run to clear run history and train from scratch.")
            print("Use --force to allow checkpoint overwrite.")
            sys.exit(1)

    # GPU preflight
    check_gpu_preflight(container)

    # Verifica se Kokoro está rodando
    check_kokoro_running()

    # Reset run se solicitado
    if args.reset_run:
        reset_run(container, args.run_name)

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    # Monta comando de treinamento
    cmd = (
        f'python3 train.py '
        f'--config "{CONFIG}" '
        f'--dataset-dir "{dataset_path}" '
        f'--run-name "{args.run_name}" '
        f'--batch-size {args.batch_size} '
        f'--max-steps {args.max_steps} '
        f'--max-epochs {args.max_epochs} '
        f'--save-every {args.save_every} '
        f'--num-workers {args.num_workers} '
        f'--gpu {args.gpu}'
    )

    try:
        run_in_container(container, cmd, batch_size=args.batch_size)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            sys.exit(1)
        raise

    # Exporta checkpoint se solicitado
    if not args.no_export:
        try:
            export_checkpoint(container, args.run_name, args.force)
        except RuntimeError as e:
            print(f"\nWarning: Failed to export checkpoint: {e}")
            print("You can manually export with:")
            print(f"  docker exec seedvc cp /app/runs/{args.run_name}/ft_model.pth /data/checkpoints/{args.run_name}.pth")
            print(f"  docker exec seedvc cp {CONFIG} /data/checkpoints/{args.run_name}.yml")
            sys.exit(1)

    # Resumo final
    print("\n" + "=" * 60)
    print("Training completed")
    print("=" * 60)
    print(f"\nRun:\n  {RUNS_DIR_CONTAINER}/{args.run_name}")

    if not args.no_export:
        print(f"\nPublished checkpoint:\n  {CHECKPOINTS_DIR_CONTAINER}/{args.run_name}.pth")
        print(f"\nPublished config:\n  {CHECKPOINTS_DIR_CONTAINER}/{args.run_name}.yml")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning do Seed-VC para voice conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Treinar dataset gold_v1
  python3 train_voice.py --dataset gold_v1 --run-name bernardo_gold_v1

  # Treinar dataset full_v1
  python3 train_voice.py --dataset full_v1 --run-name bernardo_full_v1

  # Com parâmetros customizados
  python3 train_voice.py --dataset gold_v1 --run-name test_v1 --batch-size 1 --max-steps 500

  # Retreinar sobrescrevendo checkpoint existente
  python3 train_voice.py --dataset gold_v1 --run-name bernardo_gold_v1 --force
"""
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Nome do dataset (ex: gold_v1, full_v1). Deve existir em /data/train_datasets/<dataset>/"
    )

    parser.add_argument(
        "--run-name",
        required=True,
        help="Nome da run (ex: bernardo_gold_v1). Usado para nomear o checkpoint final."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_TRAINING_PROFILE["batch_size"],
        help=f"Batch size (default: {DEFAULT_TRAINING_PROFILE['batch_size']} para RTX 2060 12GB)"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_TRAINING_PROFILE["max_steps"],
        help=f"Máximo de optimizer steps (default: {DEFAULT_TRAINING_PROFILE['max_steps']})"
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_TRAINING_PROFILE["max_epochs"],
        help=f"Máximo de epochs (default: {DEFAULT_TRAINING_PROFILE['max_epochs']})"
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_TRAINING_PROFILE["save_every"],
        help=f"Salvar checkpoint a cada N steps (default: {DEFAULT_TRAINING_PROFILE['save_every']})"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_TRAINING_PROFILE["num_workers"],
        help=f"DataLoader workers (default: {DEFAULT_TRAINING_PROFILE['num_workers']})"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=DEFAULT_TRAINING_PROFILE["gpu"],
        help=f"GPU ID (default: {DEFAULT_TRAINING_PROFILE['gpu']})"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing checkpoint in /data/checkpoints/"
    )

    parser.add_argument(
        "--reset-run",
        action="store_true",
        help="Delete run history before training (trains from scratch)"
    )

    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not export checkpoint to /data/checkpoints/ after training"
    )

    args = parser.parse_args()

    try:
        train(args)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("You can resume with the same command.")
        sys.exit(1)


if __name__ == "__main__":
    main()
