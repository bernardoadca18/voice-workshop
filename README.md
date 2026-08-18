# Voice Workshop

**English** · [Português (Brasil)](README.pt-BR.md)

Voice Workshop is a GPU-accelerated, self-hosted pipeline for producing custom-voice narration. It combines Kokoro text-to-speech, silence-aware audio segmentation, Seed-VC voice conversion, smooth chunk reconstruction, and True Peak-safe final loudness processing behind an asynchronous FastAPI service.

> Voice cloning and synthetic speech are sensitive technologies. Use only voices and recordings for which you have the necessary consent and rights, and disclose synthetic audio where applicable.

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [API](#api)
- [Models and voice references](#models-and-voice-references)
- [Configuration](#configuration)
- [Audio processing](#audio-processing)
- [Training](#training)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Operations](#operations)
- [Troubleshooting](#troubleshooting)
- [Limitations and production notes](#limitations-and-production-notes)

## Features

- End-to-end text-to-custom-voice narration.
- Kokoro TTS with optional bearer-token authentication.
- A pronunciation glossary applied only to the synthesis copy of the text.
- Audio segmentation at natural silence points instead of text splitting.
- Configurable **draft** and **final** Seed-VC profiles.
- Fine-tuned multi-checkpoint support through **run_name**.
- Zero-shot Seed-VC inference without a local fine-tuned checkpoint.
- Persistent model and voice-reference conditioning caches.
- Per-chunk level alignment and smooth crossfade reconstruction.
- Final integrated-loudness normalization constrained by real True Peak.
- Constant linear gain only: no limiter, compressor, EQ, denoising, or saturation.
- Transactional WAV output with saved-file reanalysis before replacement.
- Asynchronous jobs with serialized GPU execution.
- Docker Compose deployment and custom-voice training tooling.

## Architecture

~~~text
Client
  │ HTTP :9000
  ▼
Orchestrator (FastAPI)
  ├── pronunciation respelling
  ├── Kokoro request ─────────────────────────► Kokoro TTS :8880
  ├── silence-aware chunking
  ├── per-chunk conversion requests ──────────► Seed-VC :8001
  ├── crossfade reconstruction
  ├── LUFS + True Peak-safe post-processing
  └── asynchronous job/result API

Shared ./data
  ├── reference audio and training datasets
  ├── training runs and published checkpoints
  ├── temporary request workspaces
  └── final WAV output
~~~

| Service | Container | Port | Responsibility |
| --- | --- | ---: | --- |
| Kokoro | **kokoro** | 8880 | Base text-to-speech synthesis |
| Seed-VC | **seedvc** | 8001 | Voice conversion and model caching |
| Orchestrator | **voice-orchestrator** | 9000 | Public API, jobs, pipeline, and final processing |

Both audio engines reserve an NVIDIA GPU. The orchestrator serializes narration jobs with a process-level lock so GPU stages do not compete for VRAM.

## Pipeline

~~~text
Text
 → synthesis-only pronunciation respelling
 → Kokoro continuous WAV
 → silence detection and 4–8 second chunks
 → Seed-VC conversion per chunk
 → chunk level alignment and 100 ms crossfades
 → concatenated WAV
 → Integrated LUFS + Sample Peak + True Peak analysis
 → peak-constrained constant gain
 → saved-file reanalysis and validation
 → final narration WAV
~~~

1. Known terms receive temporary phonetic spellings. The original script remains unchanged.
2. Kokoro generates one continuous base WAV.
3. The audio—not the text—is split around natural silences.
4. Seed-VC converts each chunk using the requested checkpoint and quality.
5. Converted chunks are aligned and reconstructed with short crossfades.
6. One constant gain is selected from the loudness target and True Peak ceiling.
7. A temporary encoded WAV is reopened, measured, and validated before atomic replacement.

## Requirements

Recommended deployment:

- Linux or WSL2.
- Docker Engine and Docker Compose v2.
- NVIDIA CUDA-capable GPU, compatible driver, and NVIDIA Container Toolkit.
- Sufficient VRAM and disk space for images, models, datasets, runs, and outputs.
- **curl** for examples and optionally **jq** for polling scripts.

Verify GPU access:

~~~bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
~~~

The Seed-VC image pins upstream commit **51383efd921027683c89e5348211d93ff12ac2a8** for reproducibility. The included training defaults target a stable 12 GB GPU profile.

## Quick start

### 1. Configure the environment

~~~bash
cp .env.example .env
~~~

Edit **.env**:

~~~dotenv
KOKORO_API_KEY=
KOKORO_VOICE=pm_alex
~~~

The API key may be left empty if the Kokoro service does not require it. The file is ignored by Git; never commit credentials.

### 2. Prepare voice assets

The fixed voice-reference location is:

~~~text
data/reference/voz_referencia.wav
~~~

For fine-tuned inference, publish both matching files:

~~~text
data/checkpoints/<run_name>.pth
data/checkpoints/<run_name>.yml
~~~

Alternatively, use **run_name: zero_shot**. Zero-shot mode still requires the reference WAV.

### 3. Start the stack

~~~bash
docker compose up -d --build
docker compose ps
~~~

The first Seed-VC build clones the pinned upstream repository and installs its dependencies, so it may take several minutes.

### 4. Check health

~~~bash
curl http://localhost:8880/health
curl http://localhost:8001/health
curl http://localhost:9000/health
~~~

The orchestrator should return:

~~~json
{"status":"ok"}
~~~

Interactive OpenAPI documentation is available at [http://localhost:9000/docs](http://localhost:9000/docs).

### 5. Generate narration

~~~bash
curl -X POST http://localhost:9000/narrar   -H "Content-Type: application/json"   -d '{
    "texto": "Welcome to Voice Workshop.",
    "run_name": "zero_shot",
    "qualidade": "draft"
  }'
~~~

Response:

~~~json
{
  "job_id": "c892b318729c4893a8715fdaa0b2e44b",
  "status_url": "/jobs/c892b318729c4893a8715fdaa0b2e44b"
}
~~~

Poll and download after the status becomes **concluido**:

~~~bash
curl http://localhost:9000/jobs/c892b318729c4893a8715fdaa0b2e44b
curl -o narration.wav http://localhost:9000/jobs/c892b318729c4893a8715fdaa0b2e44b/audio
~~~

## API

The public API is served at **http://localhost:9000** and currently has no authentication.

### GET /health

Returns process health:

~~~json
{"status":"ok"}
~~~

### POST /narrar

Creates an asynchronous narration job.

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| **texto** | string | Yes | — | Text to synthesize; blank input is rejected |
| **run_name** | string | No | **minha_voz_v1** | Published model name or **zero_shot** |
| **qualidade** | string | No | **final** | **draft** or **final** |

**draft** currently uses 35 Seed-VC diffusion steps; **final** uses 75.

### GET /jobs/{job_id}

Returns the in-memory job record:

~~~json
{
  "id": "c892b318729c4893a8715fdaa0b2e44b",
  "status": "rodando",
  "chunk_atual": 0,
  "total_chunks": 0,
  "resultado_path": null,
  "erro": null,
  "criado_em": "2026-08-18T12:00:00Z"
}
~~~

| Status | Meaning |
| --- | --- |
| **pendente** | Created and waiting for processing |
| **rodando** | Pipeline is running |
| **concluido** | Final WAV is available |
| **erro** | Pipeline failed; inspect **erro** and the logs |

The current schema exposes **chunk_atual** and **total_chunks**, but processing does not update them yet.

### GET /jobs/{job_id}/audio

Returns the completed result as **audio/wav**. The WAV remains under **data/final_output** after download. Unknown, unfinished, or missing results return HTTP 404.

Complete polling example:

~~~bash
JOB_ID=$(curl -s -X POST http://localhost:9000/narrar   -H "Content-Type: application/json"   -d '{"texto":"A complete example.","run_name":"zero_shot","qualidade":"draft"}'   | jq -r '.job_id')

while true; do
  RESPONSE=$(curl -s "http://localhost:9000/jobs/$JOB_ID")
  STATUS=$(printf '%s' "$RESPONSE" | jq -r '.status')
  echo "Status: $STATUS"
  case "$STATUS" in
    concluido) break ;;
    erro) printf '%s' "$RESPONSE" | jq -r '.erro'; exit 1 ;;
  esac
  sleep 5
done

curl -o narration.wav "http://localhost:9000/jobs/$JOB_ID/audio"
~~~

## Models and voice references

Every fine-tuned **run_name** must have a **.pth** checkpoint and matching **.yml** configuration in **data/checkpoints**. The orchestrator validates both before accepting the job, allowing multiple voices in one deployment.

**zero_shot** skips local checkpoint validation and loads the official pretrained Seed-VC model.

All conversions currently use **/data/reference/voz_referencia.wav**, mapped from the repository's **data/reference/voz_referencia.wav**. A clean, single-speaker, unclipped 10–30 second recording is a practical starting point.

Seed-VC caches reference conditioning. After replacing the reference without restarting, clear it:

~~~bash
curl -X POST http://localhost:8001/cache/limpar
~~~

## Configuration

Variables passed by the provided Compose file:

| Variable | Default | Purpose |
| --- | ---: | --- |
| **KOKORO_API_KEY** | empty | Optional Kokoro bearer token |
| **KOKORO_VOICE** | **pm_alex** | Base synthesis voice |
| **AUDIO_TARGET_LUFS** | **-16.0** | Final integrated-loudness target |
| **AUDIO_MAX_TRUE_PEAK_DB** | **-1.0** | Final True Peak ceiling in dBTP |
| **AUDIO_TRUE_PEAK_TOLERANCE_DB** | **0.02** | Validation tolerance |
| **AUDIO_TRUE_PEAK_SAFETY_MARGIN_DB** | **0.05** | Internal headroom below the ceiling |
| **AUDIO_MIN_ACCEPTABLE_LUFS** | **-18.0** | Lower normal-pass bound |
| **AUDIO_MAX_ACCEPTABLE_LUFS** | **-15.0** | Upper accepted bound |

Compose sets **KOKORO_URL=http://kokoro:8880/v1/audio/speech** and **SEEDVC_URL=http://seedvc:8001**.

Additional variables supported by the Python modules, but not passed through the provided Compose file by default:

| Variable | Default | Purpose |
| --- | ---: | --- |
| **SEEDVC_DIFFUSION_STEPS_DRAFT** | **35** | Draft inference steps |
| **SEEDVC_DIFFUSION_STEPS_FINAL** | **75** | Final inference steps |
| **AUDIO_TARGET_DBFS** | **-20.0** | Per-chunk pre-crossfade reference; distinct from final LUFS |

Add them to the orchestrator service environment for a custom Docker deployment.

## Audio processing

### Silence-aware chunking

| Setting | Default |
| --- | ---: |
| Minimum chunk | 4,000 ms |
| Maximum chunk | 8,000 ms |
| Minimum silence | 300 ms |
| Silence threshold | 16 dB below average |
| Minimum retained remainder | 500 ms |
| Reconstruction crossfade | 100 ms |
| Per-chunk reference | -20 dBFS |

A forced cut is logged only when a continuous passage has no suitable pause before the maximum duration.

### True Peak-safe final processing

The final stage measures Integrated LUFS with **pyloudnorm**, Sample Peak from decoded samples with NumPy, and True Peak independently with FFmpeg's **ebur128=peak=true** filter and **lavfi.r128.true_peak** metadata.

~~~text
gain_for_loudness = target_lufs - measured_lufs
processing_ceiling = true_peak_ceiling - safety_margin
gain_allowed_by_peak = processing_ceiling - measured_true_peak
applied_gain = min(gain_for_loudness, gain_allowed_by_peak)
~~~

Defaults produce a configured ceiling of **-1.00 dBTP**, an internal processing ceiling of **-1.05 dBTP**, and **0.02 dB** validation tolerance.

If the loudness target cannot be reached safely, True Peak wins and the result is **WARNING**. No limiter, compressor, clipping, or dynamic processing is introduced.

The temporary encoded WAV is reopened and reanalyzed. Sample rate, channels, WAV subtype/bit depth, and endianness are preserved and checked. Measurement, format, save, clipping, or safety failures leave the original file untouched.

| Result | Meaning |
| --- | --- |
| **PASS** | Safe peak and loudness in the accepted range |
| **WARNING** | Safe output, but peak headroom prevented the loudness target |
| **ERROR** | Invalid audio, measurement/save failure, clipping, overshoot, unsafe loudness, or format change |

## Training

Training is controlled by **orchestrator/train_voice.py** on the host and runs inside the **seedvc** container.

Dataset layout:

~~~text
data/train_datasets/
  gold_v1/
    sample_001.wav
  full_v1/
    sample_001.wav
~~~

WAV, MP3, or FLAC files must be directly inside the selected dataset directory. Transcripts are not required by this wrapper.

Start Seed-VC and optionally free GPU memory:

~~~bash
docker compose up -d seedvc
docker compose stop kokoro orchestrator
~~~

Train and publish a voice:

~~~bash
python3 orchestrator/train_voice.py   --dataset gold_v1   --run-name bernardo_gold_v1
~~~

| Option | Default | Purpose |
| --- | ---: | --- |
| **--batch-size** | 2 | Use 1 after CUDA OOM |
| **--max-steps** | 1000 | Optimizer-step limit |
| **--max-epochs** | 1000 | Epoch limit |
| **--save-every** | 100 | Checkpoint interval |
| **--num-workers** | 0 | DataLoader workers |
| **--gpu** | 0 | CUDA device |
| **--force** | off | Permit published-checkpoint replacement |
| **--reset-run** | off | Delete the selected run and start over |
| **--no-export** | off | Do not publish the final checkpoint |

An interrupted run can be resumed with the same command unless **--reset-run** is used. Successful export creates **data/checkpoints/<run_name>.pth** and **.yml**; run history remains in **data/runs/<run_name>**.

## Project structure

~~~text
voice-workshop/
├── docker-compose.yml
├── .env.example
├── README.md
├── README.pt-BR.md
├── orchestrator/
│   ├── api.py
│   ├── jobs.py
│   ├── run_pipeline.py
│   ├── audio_chunker.py
│   ├── audio_postprocess.py
│   ├── glossario_fonetico.py
│   ├── train_voice.py
│   ├── Dockerfile
│   └── requirements.txt
├── seedvc/
│   ├── server.py
│   └── Dockerfile
├── tests/
│   └── test_audio_postprocess.py
└── data/                    # ignored by Git
    ├── reference/
    ├── train_datasets/
    ├── runs/
    ├── checkpoints/
    ├── work/
    └── final_output/
~~~

## Testing

The current automated suite targets the safety-critical final audio stage. It requires Python, FFmpeg, libsndfile, and the orchestrator dependencies.

~~~bash
sudo apt-get install ffmpeg libsndfile1
python3 -m venv .venv
source .venv/bin/activate
pip install -r orchestrator/requirements.txt
python -m unittest discover -s tests -v
~~~

Coverage includes gain increase/reduction, peak-constrained warnings, real inter-sample True Peak, output reanalysis, clipping prevention, explicit measurement failure, format preservation, deterministic results, and transactional rejection of unsafe output.

## Operations

~~~bash
# Status and logs
docker compose ps
docker compose logs -f orchestrator
docker compose logs -f seedvc kokoro

# Restart
docker compose restart
docker compose restart orchestrator

# Rebuild after source changes
docker compose up -d --build orchestrator
docker compose up -d --build seedvc

# Stop
docker compose down
~~~

Source code is copied into images rather than bind-mounted, so code changes require rebuilding the affected service. **docker compose down** preserves host files under **data** and does not remove the Kokoro named volume unless **--volumes** is supplied.

Per-request files live in **data/work/<request_id>** and are normally removed in a **finally** block. Completed files remain as **data/final_output/resultado_<request_id>.wav**.

## Troubleshooting

### Missing checkpoint or config

Verify that both files exist, or choose **zero_shot**:

~~~bash
ls -lh data/checkpoints/<run_name>.pth
ls -lh data/checkpoints/<run_name>.yml
~~~

### GPU unavailable

~~~bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
docker compose logs seedvc
~~~

Check the driver and NVIDIA Container Toolkit.

### CUDA out of memory during training

Stop other GPU services and retry with **--batch-size 1**.

### Job remains running

~~~bash
docker compose logs -f orchestrator seedvc kokoro
~~~

Seed-VC requests have a 300-second timeout. Transient connections are retried up to three times with exponential backoff.

### Reference was replaced but output did not change

Clear the reference-conditioning cache or restart Seed-VC:

~~~bash
curl -X POST http://localhost:8001/cache/limpar
~~~

### Final processing fails

~~~bash
docker compose logs --tail=200 orchestrator
~~~

Logs include before/after LUFS, Sample Peak, True Peak, gain, status, and reason. The processor fails closed: it never falls back from True Peak to Sample Peak.

## Limitations and production notes

Current limitations:

- Jobs exist only in memory and disappear after an orchestrator restart.
- The design assumes one orchestrator replica.
- Narration jobs are processed serially.
- Chunk progress fields are not updated yet.
- The API has no authentication, rate limiting, quotas, or request-size limits.
- Final WAV retention and cleanup are external responsibilities.
- The voice-reference path is fixed per deployment.
- Seed-VC requests use a fixed 300-second timeout.

Before exposing the system outside a trusted network, add an authenticated TLS reverse proxy, rate limiting, request limits, audit logging, secret management, and restricted access to ports 8001 and 8880. Use a persistent queue/database if restart tolerance or multiple replicas are required. Review consent, dataset rights, model licenses, and dependency terms before distribution or commercial use.

## License

This repository currently has no project-level license file. Kokoro, Seed-VC, model weights, and other dependencies have their own licenses and terms.

---

Leia esta documentação em [Português do Brasil](README.pt-BR.md).
