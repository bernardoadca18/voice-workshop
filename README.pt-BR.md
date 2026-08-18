# Voice Workshop

[English](README.md) · **Português (Brasil)**

O Voice Workshop é um pipeline auto-hospedado e acelerado por GPU para produzir narrações com voz personalizada. Ele combina síntese de texto com Kokoro, segmentação de áudio por silêncio, conversão de voz com Seed-VC, reconstrução suave dos trechos e pós-processamento final protegido por True Peak, tudo exposto por uma API assíncrona em FastAPI.

Para a investigação e correção do esgotamento de recursos em execuções sequenciais/batch, consulte a [Auditoria de Estabilidade em Batch](BATCH_STABILITY_AUDIT.md).

> Clonagem de voz e fala sintética são tecnologias sensíveis. Use somente vozes e gravações para as quais você tenha consentimento e os direitos necessários, e sinalize o uso de áudio sintético quando aplicável.

## Sumário

- [Funcionalidades](#funcionalidades)
- [Arquitetura](#arquitetura)
- [Pipeline](#pipeline)
- [Pré-requisitos](#pré-requisitos)
- [Início rápido](#início-rápido)
- [API](#api)
- [Modelos e referências de voz](#modelos-e-referências-de-voz)
- [Configuração](#configuração)
- [Processamento de áudio](#processamento-de-áudio)
- [Treinamento](#treinamento)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Testes](#testes)
- [Operação](#operação)
- [Solução de problemas](#solução-de-problemas)
- [Limitações e produção](#limitações-e-produção)

## Funcionalidades

- Narração completa de texto para voz personalizada.
- Kokoro TTS com autenticação opcional por bearer token.
- Glossário de pronúncia aplicado somente à cópia enviada para síntese.
- Segmentação nos silêncios naturais do áudio, sem dividir o texto.
- Perfis Seed-VC configuráveis **draft** e **final**.
- Suporte a múltiplos checkpoints por meio de **run_name**.
- Inferência zero-shot sem checkpoint fine-tuned local.
- Cache persistente dos modelos e do condicionamento da referência.
- Alinhamento de nível por chunk e reconstrução com crossfade.
- Normalização final de loudness limitada por medição real de True Peak.
- Apenas ganho linear constante: sem limiter, compressor, EQ, denoise ou saturação.
- Escrita transacional do WAV, com reanálise antes da substituição.
- Jobs assíncronos com execução serializada na GPU.
- Implantação via Docker Compose e ferramenta de treinamento de vozes.

## Arquitetura

~~~text
Cliente
  │ HTTP :9000
  ▼
Orchestrator (FastAPI)
  ├── respelling de pronúncia
  ├── requisição Kokoro ──────────────────────► Kokoro TTS :8880
  ├── fatiamento por silêncio
  ├── conversão por chunk ────────────────────► Seed-VC :8001
  ├── reconstrução com crossfade
  ├── pós-processamento LUFS + True Peak
  └── API assíncrona de jobs e resultados

./data compartilhado
  ├── referência de voz e datasets
  ├── runs e checkpoints publicados
  ├── áreas temporárias por requisição
  └── WAVs finais
~~~

| Serviço | Container | Porta | Responsabilidade |
| --- | --- | ---: | --- |
| Kokoro | **kokoro** | 8880 | Síntese base de texto para voz |
| Seed-VC | **seedvc** | 8001 | Conversão de voz e cache dos modelos |
| Orchestrator | **voice-orchestrator** | 9000 | API pública, jobs, pipeline e processamento final |

Os dois motores de áudio reservam uma GPU NVIDIA. O orchestrator serializa as narrações com um lock no processo para impedir disputa de VRAM entre etapas executadas na GPU.

## Pipeline

~~~text
Texto
 → respelling apenas para síntese
 → WAV contínuo do Kokoro
 → detecção de silêncio e chunks de 4–8 segundos
 → conversão de cada chunk no Seed-VC
 → alinhamento de nível e crossfades de 100 ms
 → WAV concatenado
 → análise de Integrated LUFS + Sample Peak + True Peak
 → ganho constante limitado pelo peak
 → reanálise e validação do arquivo salvo
 → WAV final de narração
~~~

1. Termos conhecidos recebem grafia fonética temporária. O roteiro original não é alterado.
2. O Kokoro gera um único WAV base contínuo.
3. O áudio — e não o texto — é segmentado em pausas naturais.
4. O Seed-VC converte cada chunk usando o checkpoint e a qualidade solicitados.
5. Os chunks convertidos são alinhados e unidos com crossfades curtos.
6. Um único ganho constante é escolhido considerando loudness e teto de True Peak.
7. Um WAV temporário é reaberto, medido e validado antes da substituição atômica.

## Pré-requisitos

Implantação recomendada:

- Linux ou WSL2.
- Docker Engine e Docker Compose v2.
- GPU NVIDIA compatível com CUDA, driver adequado e NVIDIA Container Toolkit.
- VRAM e espaço em disco suficientes para imagens, modelos, datasets, runs e saídas.
- **curl** para os exemplos e, opcionalmente, **jq** para os scripts de polling.

Valide o acesso à GPU:

~~~bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
~~~

A imagem do Seed-VC fixa o commit upstream **51383efd921027683c89e5348211d93ff12ac2a8** para reprodutibilidade. Os defaults de treinamento foram definidos para estabilidade em uma GPU de 12 GB.

## Início rápido

### 1. Configure o ambiente

~~~bash
cp .env.example .env
~~~

Edite o arquivo **.env**:

~~~dotenv
KOKORO_API_KEY=
KOKORO_VOICE=pm_alex
~~~

A chave pode ficar vazia se o serviço Kokoro não exigir autenticação. O arquivo é ignorado pelo Git; nunca versiona credenciais.

### 2. Prepare os arquivos de voz

A referência fixa deve estar em:

~~~text
data/reference/voz_referencia.wav
~~~

Para inferência fine-tuned, publique os dois arquivos correspondentes:

~~~text
data/checkpoints/<run_name>.pth
data/checkpoints/<run_name>.yml
~~~

Como alternativa, use **run_name: zero_shot**. O modo zero-shot ainda precisa do WAV de referência.

### 3. Inicie os serviços

~~~bash
docker compose up -d --build
docker compose ps
~~~

O primeiro build do Seed-VC clona o repositório upstream fixado e instala as dependências, portanto pode levar alguns minutos.

### 4. Verifique a saúde

~~~bash
curl http://localhost:8880/health
curl http://localhost:8001/health
curl http://localhost:9000/health
~~~

Resposta esperada do orchestrator:

~~~json
{"status":"ok"}
~~~

A documentação OpenAPI interativa fica em [http://localhost:9000/docs](http://localhost:9000/docs).

### 5. Gere uma narração

~~~bash
curl -X POST http://localhost:9000/narrar   -H "Content-Type: application/json"   -d '{
    "texto": "Bem-vindo ao Voice Workshop.",
    "run_name": "zero_shot",
    "qualidade": "draft"
  }'
~~~

Resposta:

~~~json
{
  "job_id": "c892b318729c4893a8715fdaa0b2e44b",
  "status_url": "/jobs/c892b318729c4893a8715fdaa0b2e44b"
}
~~~

Consulte o job e baixe o arquivo após o status mudar para **concluido**:

~~~bash
curl http://localhost:9000/jobs/c892b318729c4893a8715fdaa0b2e44b
curl -o narracao.wav http://localhost:9000/jobs/c892b318729c4893a8715fdaa0b2e44b/audio
~~~

## API

A API pública está em **http://localhost:9000** e atualmente não possui autenticação.

### GET /health

Retorna a saúde do processo:

~~~json
{"status":"ok"}
~~~

### POST /narrar

Cria um job assíncrono de narração.

| Campo | Tipo | Obrigatório | Default | Descrição |
| --- | --- | --- | --- | --- |
| **texto** | string | Sim | — | Texto a sintetizar; vazio é rejeitado |
| **run_name** | string | Não | **minha_voz_v1** | Modelo publicado ou **zero_shot** |
| **qualidade** | string | Não | **final** | **draft** ou **final** |

Atualmente, **draft** usa 35 diffusion steps no Seed-VC e **final** usa 75.

### GET /jobs/{job_id}

Retorna o registro mantido em memória:

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

| Status | Significado |
| --- | --- |
| **pendente** | Criado e aguardando processamento |
| **rodando** | Pipeline em execução |
| **concluido** | WAV final disponível |
| **erro** | Falha; consulte **erro** e os logs |

O schema expõe **chunk_atual** e **total_chunks**, mas a implementação ainda não atualiza esses campos durante o processamento.

### GET /jobs/{job_id}/audio

Retorna o resultado como **audio/wav**. O arquivo continua em **data/final_output** depois do download. Jobs desconhecidos, incompletos ou sem arquivo retornam HTTP 404.

Fluxo completo:

~~~bash
JOB_ID=$(curl -s -X POST http://localhost:9000/narrar   -H "Content-Type: application/json"   -d '{"texto":"Exemplo completo.","run_name":"zero_shot","qualidade":"draft"}'   | jq -r '.job_id')

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

curl -o narracao.wav "http://localhost:9000/jobs/$JOB_ID/audio"
~~~

## Modelos e referências de voz

Todo **run_name** fine-tuned precisa de um checkpoint **.pth** e de uma configuração **.yml** correspondente em **data/checkpoints**. O orchestrator valida os dois antes de aceitar o job, permitindo múltiplas vozes na mesma implantação.

O **zero_shot** ignora a validação de checkpoint local e carrega o modelo oficial pré-treinado do Seed-VC.

Todas as conversões usam atualmente **/data/reference/voz_referencia.wav**, mapeado do arquivo **data/reference/voz_referencia.wav** no repositório. Como ponto de partida, use uma gravação limpa, sem clipping, com um único locutor e aproximadamente 10–30 segundos.

O condicionamento da referência fica em cache. Se substituir o arquivo sem reiniciar o Seed-VC, limpe o cache:

~~~bash
curl -X POST http://localhost:8001/cache/limpar
~~~

## Configuração

Variáveis encaminhadas pelo Compose fornecido:

| Variável | Default | Finalidade |
| --- | ---: | --- |
| **KOKORO_API_KEY** | vazio | Bearer token opcional do Kokoro |
| **KOKORO_VOICE** | **pm_alex** | Voz base da síntese |
| **AUDIO_TARGET_LUFS** | **-16.0** | Alvo de loudness integrado final |
| **AUDIO_MAX_TRUE_PEAK_DB** | **-1.0** | Teto final em dBTP |
| **AUDIO_TRUE_PEAK_TOLERANCE_DB** | **0.02** | Tolerância de validação |
| **AUDIO_TRUE_PEAK_SAFETY_MARGIN_DB** | **0.05** | Margem interna abaixo do teto |
| **AUDIO_MIN_ACCEPTABLE_LUFS** | **-18.0** | Limite inferior para PASS normal |
| **AUDIO_MAX_ACCEPTABLE_LUFS** | **-15.0** | Limite superior aceitável |

O Compose define **KOKORO_URL=http://kokoro:8880/v1/audio/speech** e **SEEDVC_URL=http://seedvc:8001**.

Variáveis suportadas pelos módulos Python, mas não encaminhadas pelo Compose padrão:

| Variável | Default | Finalidade |
| --- | ---: | --- |
| **SEEDVC_DIFFUSION_STEPS_DRAFT** | **35** | Steps do perfil draft |
| **SEEDVC_DIFFUSION_STEPS_FINAL** | **75** | Steps do perfil final |
| **AUDIO_TARGET_DBFS** | **-20.0** | Referência por chunk antes do crossfade; diferente do LUFS final |

Para personalizá-las em Docker, adicione-as ao bloco **environment** do orchestrator.

## Processamento de áudio

### Fatiamento por silêncio

| Ajuste | Default |
| --- | ---: |
| Chunk mínimo | 4.000 ms |
| Chunk máximo | 8.000 ms |
| Silêncio mínimo | 300 ms |
| Threshold de silêncio | 16 dB abaixo da média |
| Sobra mínima mantida | 500 ms |
| Crossfade | 100 ms |
| Referência por chunk | -20 dBFS |

Um corte forçado só ocorre — e é registrado no log — quando um trecho contínuo não tem uma pausa adequada antes do limite máximo.

### Pós-processamento final protegido por True Peak

A etapa final mede Integrated LUFS com **pyloudnorm**, Sample Peak diretamente das amostras com NumPy e True Peak de forma independente com o filtro **ebur128=peak=true** do FFmpeg e os metadados **lavfi.r128.true_peak**.

~~~text
ganho_para_loudness = target_lufs - measured_lufs
teto_de_processamento = teto_true_peak - safety_margin
ganho_permitido_pelo_peak = teto_de_processamento - measured_true_peak
ganho_aplicado = min(ganho_para_loudness, ganho_permitido_pelo_peak)
~~~

Os defaults resultam em teto configurado de **-1,00 dBTP**, teto interno de processamento de **-1,05 dBTP** e tolerância de validação de **0,02 dB**.

Se não for possível chegar ao loudness desejado com segurança, o True Peak tem prioridade e o resultado é **WARNING**. Nenhum limiter, compressor, clipping ou processamento dinâmico é introduzido.

O WAV temporário codificado é reaberto e reanalisado. Sample rate, canais, subtype/bit depth e endianness são preservados e conferidos. Falhas de medição, formato, salvamento, clipping ou segurança deixam o arquivo original intocado.

| Resultado | Significado |
| --- | --- |
| **PASS** | Peak seguro e loudness na faixa |
| **WARNING** | Saída segura, mas o teto impediu atingir o target |
| **ERROR** | Áudio inválido, falha de medição/salvamento, clipping, overshoot, loudness inseguro ou mudança de formato |

## Treinamento

O treinamento é controlado por **orchestrator/train_voice.py** no host e executado dentro do container **seedvc**.

Estrutura dos datasets:

~~~text
data/train_datasets/
  gold_v1/
    sample_001.wav
  full_v1/
    sample_001.wav
~~~

Os arquivos WAV, MP3 ou FLAC devem ficar diretamente no diretório escolhido. Este wrapper não exige transcrições.

Inicie o Seed-VC e, opcionalmente, libere VRAM:

~~~bash
docker compose up -d seedvc
docker compose stop kokoro orchestrator
~~~

Treine e publique uma voz:

~~~bash
python3 orchestrator/train_voice.py   --dataset gold_v1   --run-name bernardo_gold_v1
~~~

| Opção | Default | Finalidade |
| --- | ---: | --- |
| **--batch-size** | 2 | Use 1 após CUDA OOM |
| **--max-steps** | 1000 | Limite de optimizer steps |
| **--max-epochs** | 1000 | Limite de epochs |
| **--save-every** | 100 | Intervalo de checkpoints |
| **--num-workers** | 0 | Workers do DataLoader |
| **--gpu** | 0 | Dispositivo CUDA |
| **--force** | desligado | Permite substituir checkpoint publicado |
| **--reset-run** | desligado | Apaga a run escolhida e recomeça |
| **--no-export** | desligado | Não publica o checkpoint final |

Uma run interrompida pode ser retomada com o mesmo comando, salvo quando **--reset-run** for usado. O export cria **data/checkpoints/<run_name>.pth** e **.yml**; o histórico permanece em **data/runs/<run_name>**.

## Estrutura do projeto

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
└── data/                    # ignorado pelo Git
    ├── reference/
    ├── train_datasets/
    ├── runs/
    ├── checkpoints/
    ├── work/
    └── final_output/
~~~

## Testes

A suíte atual cobre a etapa crítica de pós-processamento final e precisa de Python, FFmpeg, libsndfile e dependências do orchestrator.

~~~bash
sudo apt-get install ffmpeg libsndfile1
python3 -m venv .venv
source .venv/bin/activate
pip install -r orchestrator/requirements.txt
python -m unittest discover -s tests -v
~~~

A cobertura inclui aumento e redução de ganho, WARNING limitado por peak, True Peak inter-sample real, reanálise do output, prevenção de clipping, falha explícita de medição, preservação de formato, determinismo e rejeição transacional de saída insegura.

## Operação

~~~bash
# Status e logs
docker compose ps
docker compose logs -f orchestrator
docker compose logs -f seedvc kokoro

# Reinício
docker compose restart
docker compose restart orchestrator

# Rebuild após mudanças no código
docker compose up -d --build orchestrator
docker compose up -d --build seedvc

# Encerrar
docker compose down
~~~

O código é copiado para as imagens e não montado por bind; alterações exigem rebuild do serviço afetado. **docker compose down** preserva os arquivos em **data** e não remove o volume nomeado do Kokoro sem **--volumes**.

Os intermediários ficam em **data/work/<request_id>** e normalmente são removidos em um bloco **finally**. As saídas permanecem em **data/final_output/resultado_<request_id>.wav**.

## Solução de problemas

### Checkpoint ou config ausente

Confira os dois arquivos ou use **zero_shot**:

~~~bash
ls -lh data/checkpoints/<run_name>.pth
ls -lh data/checkpoints/<run_name>.yml
~~~

### GPU indisponível

~~~bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
docker compose logs seedvc
~~~

Verifique o driver e o NVIDIA Container Toolkit.

### CUDA OOM no treinamento

Pare os outros serviços de GPU e repita com **--batch-size 1**.

### Job permanece em rodando

~~~bash
docker compose logs -f orchestrator seedvc kokoro
~~~

As chamadas ao Seed-VC têm timeout de 300 segundos. Erros transitórios de conexão são tentados até três vezes com backoff exponencial.

### Referência substituída sem mudar o resultado

Limpe o cache ou reinicie o Seed-VC:

~~~bash
curl -X POST http://localhost:8001/cache/limpar
~~~

### Falha no pós-processamento

~~~bash
docker compose logs --tail=200 orchestrator
~~~

Os logs mostram LUFS, Sample Peak, True Peak, ganho, status e motivo antes/depois. O processador falha de forma segura e nunca usa Sample Peak como fallback de True Peak.

## Limitações e produção

Limitações atuais:

- Jobs ficam apenas em memória e somem quando o orchestrator reinicia.
- A arquitetura pressupõe uma única réplica do orchestrator.
- As narrações são processadas em série.
- Os campos de progresso por chunk ainda não são atualizados.
- A API não tem autenticação, rate limiting, quotas ou limite de request.
- Retenção e limpeza dos WAVs finais são responsabilidades externas.
- O caminho da referência de voz é fixo por implantação.
- As chamadas ao Seed-VC usam timeout fixo de 300 segundos.

Antes de expor o sistema fora de uma rede confiável, adicione proxy reverso autenticado com TLS, rate limiting, limites de requisição, auditoria, gerenciamento de secrets e restrição das portas 8001 e 8880. Use fila/banco persistente se precisar tolerar reinícios ou múltiplas réplicas. Revise consentimento, direitos dos datasets, licenças dos modelos e termos das dependências antes de distribuir ou usar comercialmente.

## Licença

Este repositório não possui atualmente uma licença própria. Kokoro, Seed-VC, pesos de modelos e outras dependências têm licenças e termos próprios.

---

Read this documentation in [English](README.md).
