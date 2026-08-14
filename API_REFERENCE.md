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
| run_name | string | Não | Nome do modelo treinado. Deve ter config correspondente em `/data/checkpoints/{run_name}.yml` |
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
  "detail": "Checkpoint não encontrado: /data/checkpoints/ft_model.pth"
}
```

**Exemplo:**
```bash
curl -X POST http://localhost:9000/narrar \
  -H "Content-Type: application/json" \
  -d '{
    "texto": "Olá, este é um exemplo de narração.",
    "run_name": "minha_voz_v1",
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

## Fluxo de Uso Típico

```bash
# 1. Criar job
RESPONSE=$(curl -s -X POST http://localhost:9000/narrar \
  -H "Content-Type: application/json" \
  -d '{"texto": "Texto de exemplo", "qualidade": "draft"}')

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

async function criarNarracao(texto, qualidade = 'draft') {
  // Criar job
  const { data: { job_id } } = await axios.post('http://localhost:9000/narrar', {
    texto,
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
  local qualidade="${2:-draft}"
  local output="${3:-audio.wav}"
  
  # Criar job
  local job_id=$(curl -s -X POST http://localhost:9000/narrar \
    -H "Content-Type: application/json" \
    -d "{\"texto\": \"$texto\", \"qualidade\": \"$qualidade\"}" \
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
criar_narracao "Olá, este é um exemplo de narração." draft minha_narracao.wav
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

**Versão da API**: 0.1.0
**Última atualização**: 2026-08-14
