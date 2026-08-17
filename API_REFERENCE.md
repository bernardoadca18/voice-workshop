# API Reference - Voice Workshop

## Base URL

```
http://localhost:9000
```

## Authentication

Atualmente sem autenticação. Em produção, considere adicionar API key.

---

## Endpoints

### Health Check

Verifica se o serviço está rodando.

```http
GET /health
```

**Response 200:**
```json
{
  "status": "ok"
}
```

---

### Criar Narração

Inicia uma narração de forma assíncrona. Retorna imediatamente com um job_id.

```http
POST /narrar
Content-Type: application/json
```

**Request Body:**
```json
{
  "texto": "string (obrigatório)",
  "run_name": "string (default: minha_voz_v1)",
  "qualidade": "draft | final (default: final)"
}
```

**Parâmetros:**

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| texto | string | Sim | Texto a ser narrado. Não pode estar vazio. |
| run_name | string | Não | Nome do modelo treinado. Deve ter checkpoint e config correspondentes em `/data/checkpoints/{run_name}.pth` e `/data/checkpoints/{run_name}.yml`. Use "zero_shot" para modelo pretrained sem fine-tuning. |
| qualidade | string | Não | "draft" para previews rápidos (25 passos), "final" para qualidade máxima (45 passos) |

**Response 200:**
```json
{
  "job_id": "c892b318729c4893a8715fdaa0b2e44b",
  "status_url": "/jobs/c892b318729c4893a8715fdaa0b2e44b"
}
```

**Response 400:**
```json
{
  "detail": "Campo 'texto' vazio"
}
```
ou
```json
{
  "detail": "Checkpoint não encontrado: /data/checkpoints/bernardo_gold_v1.pth\nCertifique-se de treinar o modelo 'bernardo_gold_v1' antes de usá-lo na inferência."
}
```

**Exemplo:**
```bash
curl -X POST http://localhost:9000/narrar \
  -H "Content-Type: application/json" \
  -d '{
    "texto": "Olá, este é um exemplo de narração.",
    "run_name": "bernardo_gold_v1",
    "qualidade": "draft"
  }'
```

---

### Consultar Status do Job

Consulta o status atual de um job de narração.

```http
GET /jobs/{job_id}
```

**Path Parameters:**

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| job_id | string | ID do job retornado pelo endpoint `/narrar` |

**Response 200:**
```json
{
  "id": "c892b318729c4893a8715fdaa0b2e44b",
  "status": "rodando",
  "chunk_atual": 0,
  "total_chunks": 0,
  "resultado_path": null,
  "erro": null,
  "criado_em": "2026-08-14T14:15:55.894874Z"
}
```

**Status Possíveis:**

| Status | Descrição |
|--------|-----------|
| pendente | Job aguardando processamento |
| rodando | Job em processamento |
| concluido | Job finalizado com sucesso |
| erro | Job falhou (ver campo `erro`) |

**Campos da Resposta:**

| Campo | Tipo | Descrição |
|-------|------|-----------|
| id | string | ID único do job |
| status | string | Status atual (pendente/rodando/concluido/erro) |
| chunk_atual | integer | Chunk sendo processado (seed-vc) |
| total_chunks | integer | Total de chunks a processar |
| resultado_path | string \| null | Caminho do arquivo de áudio (quando concluído) |
| erro | string \| null | Mensagem de erro (quando status=erro) |
| criado_em | string | Timestamp de criação (ISO 8601) |

**Response 404:**
```json
{
  "detail": "Job não encontrado"
}
```

**Exemplo:**
```bash
curl http://localhost:9000/jobs/c892b318729c4893a8715fdaa0b2e44b
```

---

### Baixar Áudio

Baixa o áudio de um job concluído. O arquivo é removido automaticamente após o download.

```http
GET /jobs/{job_id}/audio
```

**Path Parameters:**

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| job_id | string | ID do job concluído |

**Response 200:**
```
Content-Type: audio/wav
Content-Disposition: attachment; filename="resultado_xxx.wav"

[binary audio data]
```

**Response 404:**
```json
{
  "detail": "Job não encontrado"
}
```
ou
```json
{
  "detail": "Áudio ainda não disponível"
}
```
ou
```json
{
  "detail": "Áudio não disponível"
}
```

**Notas:**
- O arquivo é removido após o download
- Job deve ter status "concluido"
- Formato: WAV, 22kHz, mono ou estéreo

**Exemplo:**
```bash
curl -o audio.wav http://localhost:9000/jobs/c892b318729c4893a8715fdaa0b2e44b/audio
```

---

## Múltiplos Modelos (Multi-Checkpoint)

O Voice Workshop suporta múltiplos modelos treinados, cada um identificado por um `run_name`.

### Estrutura de Checkpoints

```
data/checkpoints/
├── bernardo_gold_v1.pth    # Modelo treinado com dataset gold_v1
├── bernardo_gold_v1.yml    # Config do modelo
├── bernardo_full_v1.pth    # Modelo treinado com dataset full_v1
└── bernardo_full_v1.yml    # Config do modelo
```

### Usando Diferentes Modelos

```bash
# Usar modelo gold_v1
curl -X POST http://localhost:9000/narrar \
  -H "Content-Type: application/json" \
  -d '{
    "texto": "Texto narrado com modelo gold.",
    "run_name": "bernardo_gold_v1",
    "qualidade": "final"
  }'

# Usar modelo full_v1
curl -X POST http://localhost:9000/narrar \
  -H "Content-Type: application/json" \
  -d '{
    "texto": "Texto narrado com modelo full.",
    "run_name": "bernardo_full_v1",
    "qualidade": "final"
  }'
```

### Modo Zero-Shot

Use `run_name="zero_shot"` para usar o modelo pretrained oficial do Seed-VC sem fine-tuning:

```bash
curl -X POST http://localhost:9000/narrar \
  -H "Content-Type: application/json" \
  -d '{
    "texto": "Texto sem fine-tuning.",
    "run_name": "zero_shot",
    "qualidade": "draft"
  }'
```

---

## Treinamento de Modelos

### Treinar Novo Modelo

```bash
# Treinar dataset gold_v1
python3 orchestrator/train_voice.py \
  --dataset gold_v1 \
  --run-name bernardo_gold_v1

# Treinar dataset full_v1
python3 orchestrator/train_voice.py \
  --dataset full_v1 \
  --run-name bernardo_full_v1
```

Após o treinamento, o checkpoint será automaticamente publicado em:
- `/data/checkpoints/{run_name}.pth`
- `/data/checkpoints/{run_name}.yml`

### Parâmetros de Treinamento

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `--dataset` | obrigatório | Nome do dataset (gold_v1, full_v1) |
| `--run-name` | obrigatório | Nome do modelo/checkpoint final |
| `--batch-size` | 2 | Batch size (use 1 se OOM) |
| `--max-steps` | 1000 | Máximo de optimizer steps |
| `--max-epochs` | 1000 | Máximo de epochs |
| `--save-every` | 100 | Salvar checkpoint a cada N steps |
| `--num-workers` | 0 | DataLoader workers |
| `--gpu` | 0 | GPU ID |
| `--force` | - | Permitir sobrescrever checkpoint existente em /data/checkpoints/ |
| `--reset-run` | - | Deletar histórico da run antes de treinar (treina do zero) |
| `--no-export` | - | Não exportar checkpoint após treino |

---

## Fluxo de Uso Típico

```bash
# 1. Criar job
RESPONSE=$(curl -s -X POST http://localhost:9000/narrar \
  -H "Content-Type: application/json" \
  -d '{"texto": "Texto de exemplo", "run_name": "bernardo_gold_v1", "qualidade": "draft"}')

JOB_ID=$(echo $RESPONSE | jq -r '.job_id')

# 2. Polling até conclusão
while true; do
  STATUS=$(curl -s http://localhost:9000/jobs/$JOB_ID | jq -r '.status')
  
  case $STATUS in
    concluido) break ;;
    erro) 
      echo "Erro: $(curl -s http://localhost:9000/jobs/$JOB_ID | jq -r '.erro')"
      exit 1
      ;;
  esac
  
  sleep 5
done

# 3. Baixar áudio
curl -o audio.wav http://localhost:9000/jobs/$JOB_ID/audio
```

---

## Códigos de Erro Comuns

### 400 Bad Request

- **Texto vazio**: Campo `texto` está vazio ou contém apenas espaços
- **Run name inválido**: Checkpoint ou config não encontrado para o `run_name` especificado
- **Qualidade inválida**: Valor diferente de "draft" ou "final"

### 404 Not Found

- **Job não encontrado**: job_id não existe na memória do orchestrator
- **Áudio não disponível**: Job ainda não concluído ou já foi baixado

### 500 Internal Server Error

- Erro interno no processamento do pipeline
- Verificar logs: `docker logs voice-orchestrator`

---

## Limitações Conhecidas

1. **Processamento Serial**: Apenas uma narração por vez (lock de GPU)
2. **Jobs em Memória**: Jobs são perdidos se o orchestrator reiniciar
3. **Arquivo Temporário**: Áudio é removido após primeiro download
4. **Timeout**: Jobs muito longos podem exceder timeout do Seed-VC (300s)

---

## Exemplos por Linguagem

### Python

```python
import requests
import time

# Criar narração
response = requests.post(
    "http://localhost:9000/narrar",
    json={
        "texto": "Olá, este é um teste.",
        "run_name": "bernardo_gold_v1",
        "qualidade": "draft"
    }
)
job_id = response.json()["job_id"]

# Aguardar conclusão
while True:
    status = requests.get(f"http://localhost:9000/jobs/{job_id}").json()
    if status["status"] == "concluido":
        break
    elif status["status"] == "erro":
        raise Exception(status["erro"])
    time.sleep(5)

# Baixar áudio
audio_response = requests.get(f"http://localhost:9000/jobs/{job_id}/audio")
with open("audio.wav", "wb") as f:
    f.write(audio_response.content)
```

### JavaScript (Node.js)

```javascript
const axios = require('axios');

async function criarNarracao(texto, runName = 'bernardo_gold_v1', qualidade = 'draft') {
  // Criar job
  const { data: { job_id } } = await axios.post('http://localhost:9000/narrar', {
    texto,
    run_name: runName,
    qualidade
  });

  // Aguardar conclusão
  while (true) {
    const { data: status } = await axios.get(`http://localhost:9000/jobs/${job_id}`);
    
    if (status.status === 'concluido') break;
    if (status.status === 'erro') throw new Error(status.erro);
    
    await new Promise(resolve => setTimeout(resolve, 5000));
  }

  // Baixar áudio
  const response = await axios.get(`http://localhost:9000/jobs/${job_id}/audio`, {
    responseType: 'arraybuffer'
  });
  
  return response.data;
}

// Uso
criarNarracao('Olá, este é um teste.').then(audio => {
  require('fs').writeFileSync('audio.wav', audio);
});
```

### Bash Script

```bash
#!/bin/bash

criar_narracao() {
  local texto="$1"
  local run_name="${2:-bernardo_gold_v1}"
  local qualidade="${3:-draft}"
  local output="${4:-audio.wav}"
  
  # Criar job
  local job_id=$(curl -s -X POST http://localhost:9000/narrar \
    -H "Content-Type: application/json" \
    -d "{\"texto\": \"$texto\", \"run_name\": \"$run_name\", \"qualidade\": \"$qualidade\"}" \
    | jq -r '.job_id')
  
  echo "Job criado: $job_id"
  
  # Aguardar conclusão
  while true; do
    local status=$(curl -s http://localhost:9000/jobs/$job_id | jq -r '.status')
    
    case $status in
      concluido) break ;;
      erro)
        echo "Erro: $(curl -s http://localhost:9000/jobs/$job_id | jq -r '.erro')"
        return 1
        ;;
    esac
    
    echo "Status: $status"
    sleep 5
  done
  
  # Baixar áudio
  curl -s -o "$output" http://localhost:9000/jobs/$job_id/audio
  echo "Áudio salvo em: $output"
}

# Uso
criar_narracao "Olá, este é um exemplo de narração." bernardo_gold_v1 draft minha_narracao.wav
```

---

## Integração com Outras Ferramentas

### FFmpeg - Converter para MP3

```bash
curl -o audio.wav http://localhost:9000/jobs/{job_id}/audio
ffmpeg -i audio.wav -codec:a libmp3lame -qscale:a 2 audio.mp3
```

### Whisper - Transcrever

```bash
# Após baixar o áudio
whisper audio.wav --language pt --model medium
```

### YouTube Upload

```bash
# Usar YouTube Data API ou ferramentas como youtube-upload
youtube-upload --title="Minha Narração" audio.wav
```

---

**Versão da API**: 0.2.0
**Última atualização**: 2026-08-17
