# Voice Workshop - Guia Completo

## Visão Geral

O **Voice Workshop** é um pipeline de síntese de voz que converte texto em narração com voz personalizada. O sistema usa três componentes principais:

1. **Kokoro** - Sintetizador de voz TTS (Text-to-Speech) em português
2. **Seed-VC** - Conversor de voz que aplica características de uma voz de referência
3. **Orchestrator** - API HTTP que coordena todo o pipeline

## Arquitetura

```
Texto → Kokoro (TTS) → Fatiamento por Silêncio → Seed-VC (Voice Conversion) → Concatenação → Normalização → Áudio Final
```

### Fluxo Detalhado

1. **Síntese (Kokoro)**: O texto é convertido em áudio contínuo usando síntese TTS
2. **Fatiamento Inteligente**: O áudio é dividido em chunks de 4-8 segundos nos pontos de silêncio natural
3. **Conversão de Voz (Seed-VC)**: Cada chunk é convertido para a voz de referência usando um modelo fine-tuned
4. **Concatenação**: Os chunks convertidos são unidos com crossfade suave
5. **Normalização**: O áudio final é normalizado para loudness padrão de YouTube/streaming (-16 LUFS)

## Componentes

### 1. Kokoro Server (Porta 8880)

- **Imagem**: `hwdsl2/kokoro-server:cuda`
- **Função**: Sintetizador TTS em português
- **Endpoint**: `POST /v1/audio/speech`
- **Voice padrão**: `pm_alex`

### 2. Seed-VC Server (Porta 8001)

- **Função**: Conversão de voz usando modelo fine-tuned
- **Endpoint**: `POST /convert`
- **Checkpoints**: `/data/checkpoints/{run_name}.pth`
- **Config por voz**: `/data/checkpoints/{run_name}.yml`
- **Cache inteligente**: Mantém modelo e conditioning de referência em memória

### 3. Orchestrator API (Porta 9000)

- **Função**: Coordena o pipeline completo
- **Endpoints**:
  - `GET /health` - Status do serviço
  - `POST /narrar` - Inicia narração assíncrona
  - `GET /jobs/{job_id}` - Consulta status do job
  - `GET /jobs/{job_id}/audio` - Baixa áudio concluído

## Estrutura de Diretórios

```
data/
├── train_datasets/           # Datasets de treinamento (múltiplos)
│   ├── gold_v1/              # Dataset gold (31 WAVs)
│   │   └── *.wav
│   └── full_v1/              # Dataset full (56 WAVs)
│       └── *.wav
│
├── runs/                     # Runs de treinamento (persistente)
│   ├── bernardo_gold_v1/
│   │   ├── ft_model.pth
│   │   ├── config_dit_mel_seed_uvit_whisper_small_wavenet.yml
│   │   └── checkpoints intermediários
│   └── bernardo_full_v1/
│       └── ...
│
├── checkpoints/              # Modelos publicados (multi-checkpoint)
│   ├── bernardo_gold_v1.pth
│   ├── bernardo_gold_v1.yml
│   ├── bernardo_full_v1.pth
│   └── bernardo_full_v1.yml
│
├── reference/
│   └── voz_referencia.wav    # Amostra de voz de referência (10-30s)
│
├── final_output/             # Áudios finais
└── work/                     # Arquivos temporários
```

## Treinamento de Modelos

### Múltiplos Datasets e Checkpoints

O sistema suporta múltiplos datasets de treinamento e múltiplos checkpoints publicados:

- **gold_v1**: Dataset selecionado (31 WAVs de alta qualidade)
- **full_v1**: Dataset completo (56 WAVs)

Cada treinamento gera um checkpoint independente identificado pelo `run_name`.

### Perfil de Treinamento RTX 2060 12GB

O perfil padrão é otimizado para estabilidade em GPU com 12GB VRAM:

```
batch_size = 2
max_steps = 1000
max_epochs = 1000
save_every = 100
num_workers = 0
gpu = 0
```

**Prioridade**: estabilidade > evitar CUDA OOM > reprodutibilidade > velocidade

### Treinar Dataset Gold

```bash
python3 orchestrator/train_voice.py \
  --dataset gold_v1 \
  --run-name bernardo_gold_v1
```

### Treinar Dataset Full

```bash
python3 orchestrator/train_voice.py \
  --dataset full_v1 \
  --run-name bernardo_full_v1
```

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

### Exemplos

```bash
# Treinamento padrão
python3 orchestrator/train_voice.py --dataset gold_v1 --run-name bernardo_gold_v1

# Com batch size menor (se OOM)
python3 orchestrator/train_voice.py --dataset gold_v1 --run-name test_v1 --batch-size 1

# Retreinar do zero (deleta run antiga)
python3 orchestrator/train_voice.py --dataset gold_v1 --run-name bernardo_gold_v1 --reset-run

# Sobrescrever checkpoint existente (após treino)
python3 orchestrator/train_voice.py --dataset gold_v1 --run-name bernardo_gold_v1 --force

# Combinação: retreinar e sobrescrever checkpoint
python3 orchestrator/train_voice.py --dataset gold_v1 --run-name bernardo_gold_v1 --reset-run --force
```

### Tratamento de CUDA OOM

Se o treinamento falhar com CUDA out of memory:

```
CUDA OUT OF MEMORY DETECTED
============================================================

Current batch size: 2

Recommended retry:
  --batch-size 1

Do not change other training parameters when comparing Gold vs Full.
```

**Importante**: Não mude outros parâmetros ao comparar datasets. Use apenas `--batch-size 1` como fallback.

### Proteções de Treinamento

1. **Validação de dataset**: Verifica se dataset existe e contém arquivos de áudio
2. **Validação de run_name**: Apenas caracteres seguros (letras, números, `.`, `_`, `-`)
3. **Proteção contra overwrite**: Não sobrescreve checkpoint final sem `--force`
4. **Proteção contra dataset root**: Não aceita `train_datasets` como dataset, deve especificar um específico
5. **Resume preservado**: Treinamentos interrompidos podem ser retomados com o mesmo comando
6. **Persistência**: Runs são salvas em `./data/runs/` no host

### Zero-Shot (sem fine-tuning)

O modo `run_name="zero_shot"` usa o modelo pretrained oficial do Seed-VC sem checkpoint local:

```json
{
  "texto": "Texto de teste.",
  "run_name": "zero_shot",
  "qualidade": "final"
}
```

## Pré-requisitos

### Arquivos Necessários

```
data/
├── reference/
│   └── voz_referencia.wav      # Amostra de voz de referência (10-30s)
├── train_datasets/
│   ├── gold_v1/
│   │   └── *.wav               # 31 arquivos de áudio
│   └── full_v1/
│       └── *.wav               # 56 arquivos de áudio
└── checkpoints/
    ├── bernardo_gold_v1.pth    # Após treinamento
    └── bernardo_gold_v1.yml
```

### Variáveis de Ambiente

```bash
# Obrigatórias
KOKORO_API_KEY=sua_api_key_aqui

# Opcionais (com defaults)
KOKORO_VOICE=pm_alex                          # Voz do Kokoro
KOKORO_URL=http://kokoro:8880/v1/audio/speech
SEEDVC_URL=http://seedvc:8001
SEEDVC_DIFFUSION_STEPS_DRAFT=25               # Modo rápido
SEEDVC_DIFFUSION_STEPS_FINAL=45               # Modo qualidade
AUDIO_TARGET_LUFS=-16.0                       # Loudness alvo
AUDIO_TARGET_DBFS=-20.0                       # Normalização de chunks
```

## Instalação e Execução

### 1. Preparar Variáveis de Ambiente

```bash
# Criar arquivo .env
cat > .env << EOF
KOKORO_API_KEY=sua_api_key_aqui
EOF
```

### 2. Iniciar os Serviços

```bash
# Subir todos os containers
docker-compose up -d

# Verificar status
docker-compose ps

# Acompanhar logs
docker-compose logs -f orchestrator
```

### 3. Verificar Saúde dos Serviços

```bash
# Kokoro
curl http://localhost:8880/health
# {"status":"ok","engine":"kokoro"}

# Seed-VC
curl http://localhost:8001/health
# {"status":"ok","modelos_carregados":[]}

# Orchestrator
curl http://localhost:9000/health
# {"status":"ok"}
```

## Usando a API

### Criar uma Narração

```bash
# Requisição simples
curl -X POST http://localhost:9000/narrar \
  -H "Content-Type: application/json" \
  -d '{
    "texto": "Olá, bem-vindo ao teste do sistema de narração.",
    "run_name": "bernardo_gold_v1",
    "qualidade": "draft"
  }'

# Resposta
{
  "job_id": "c892b318729c4893a8715fdaa0b2e44b",
  "status_url": "/jobs/c892b318729c4893a8715fdaa0b2e44b"
}
```

### Parâmetros da Requisição

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|-------------|-----------|
| `texto` | string | Sim | Texto a ser narrado |
| `run_name` | string | Não | Nome do modelo treinado (default: "minha_voz_v1") |
| `qualidade` | string | Não | "draft" (rápido) ou "final" (qualidade) |

### Consultar Status do Job

```bash
curl http://localhost:9000/jobs/{job_id}

# Respostas possíveis:

# Pendente
{"id":"...", "status":"pendente", ...}

# Processando
{"id":"...", "status":"rodando", ...}

# Concluído
{
  "id": "...",
  "status": "concluido",
  "resultado_path": "/data/final_output/resultado_xxx.wav",
  ...
}

# Erro
{"id":"...", "status":"erro", "erro":"mensagem de erro", ...}
```

### Baixar o Áudio

```bash
# Baixar áudio concluído
curl -o audio_final.wav http://localhost:9000/jobs/{job_id}/audio
```

## Exemplo Completo

```bash
# 1. Enviar texto para narração
JOB_ID=$(curl -s -X POST http://localhost:9000/narrar \
  -H "Content-Type: application/json" \
  -d '{"texto": "Este é um exemplo completo de narração.", "qualidade": "draft"}' \
  | jq -r '.job_id')

echo "Job ID: $JOB_ID"

# 2. Aguardar processamento (polling)
while true; do
  STATUS=$(curl -s http://localhost:9000/jobs/$JOB_ID | jq -r '.status')
  echo "Status: $STATUS"
  
  if [ "$STATUS" = "concluido" ]; then
    break
  elif [ "$STATUS" = "erro" ]; then
    curl -s http://localhost:9000/jobs/$JOB_ID | jq -r '.erro'
    exit 1
  fi
  
  sleep 5
done

# 3. Baixar áudio
curl -o narracao.wav http://localhost:9000/jobs/$JOB_ID/audio
echo "Áudio salvo em: narracao.wav"
```

## Qualidades de Processamento

### Draft (Rápido)
- **Difusion Steps**: 25
- **Uso**: Previews, testes rápidos
- **Tempo**: ~2x mais rápido que final

### Final (Qualidade)
- **Diffusion Steps**: 45
- **Uso**: Produção, vídeos finais
- **Tempo**: Mais lento, melhor qualidade

## Processamento de Áudio

### Fatiamento por Silêncio

O sistema detecta automaticamente pontos de silêncio natural na fala para dividir o áudio em chunks de 4-8 segundos. Isso garante:

- Conversão mais estável no Seed-VC
- Preservação da prosódia natural
- Evita cortes no meio de palavras

### Glossário Fonético

Termos problemáticos são automaticamente ajustados para melhor pronúncia:

```python
# Exemplos do glossário
"kaer morhen" → "ker mórren"
"witcher" → "uícher"
"geralt" → "guéralt"
```

### Normalização de Loudness

O áudio final é normalizado para **-16 LUFS**, padrão para:
- YouTube
- Spotify
- Plataformas de streaming

## Troubleshooting

### Erro: "Checkpoint não encontrado"

```bash
# Verificar se os arquivos existem
ls -l data/checkpoints/
# Deve conter: bernardo_gold_v1.pth e bernardo_gold_v1.yml
```

### Erro: "Config não encontrado para run_name"

```bash
# Verificar se o config existe para o run_name usado
ls -l data/checkpoints/bernardo_gold_v1.yml
```

### Container Seed-VC não inicia

```bash
# Verificar logs
docker logs seedvc

# Problema comum: GPU não disponível
nvidia-smi
```

### Job fica em "rodando" indefinidamente

```bash
# Verificar logs do orchestrator
docker logs voice-orchestrator

# Verificar logs do SeedVC
docker logs seedvc
```

### Limpar Cache do Seed-VC

```bash
# Se trocar o áudio de referência, limpar o cache
curl -X POST http://localhost:8001/cache/limpar
```

## Performance

### Tempos Típicos (GPU NVIDIA)

- **Texto curto** (~10 palavras): 1-2 minutos
- **Texto médio** (~50 palavras): 3-5 minutos
- **Texto longo** (~200 palavras): 10-15 minutos

### Fatores que Afetam Performance

1. **Tamanho do texto**: Mais texto = mais chunks = mais tempo
2. **Qualidade escolhida**: draft é ~2x mais rápido
3. **Carga da GPU**: Outros processos podem impactar
4. **Primeira execução**: Modelo demora ~60s para carregar

## Manutenção

### Reiniciar Serviços

```bash
# Reiniciar tudo
docker-compose restart

# Reiniciar serviço específico
docker-compose restart orchestrator
```

### Verificar Uso de Recursos

```bash
# Uso de GPU
nvidia-smi

# Logs em tempo real
docker-compose logs -f
```

### Limpar Arquivos Temporários

```bash
# Arquivos de trabalho são limpos automaticamente
# Áudios finais ficam em: data/final_output/

# Limpar manualmente se necessário
rm -rf data/work/*
```

---

**Dica**: Sempre use qualidade "draft" para testes e "final" para produção.
