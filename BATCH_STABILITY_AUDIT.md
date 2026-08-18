# Auditoria de Estabilidade em Batch — Voice Workshop

**Data:** 18 de agosto de 2026
**Escopo:** inferência, ciclo de vida de modelos, concorrência, retries HTTP, filas, memória CPU/GPU, swap, I/O, containers, temporários e operação em batch.

## Resumo executivo

O travamento da segunda narração foi reproduzido e a causa raiz foi confirmada por telemetria e logs. Não era uma falha do pós-processamento de áudio, do SSD ou da GPU.

O problema era uma combinação de três defeitos críticos:

1. O Seed-VC mantinha todos os modelos usados em um cache sem limite.
2. O cache e o carregamento não tinham lock. Requisições concorrentes atravessavam simultaneamente a verificação do cache e carregavam várias cópias do mesmo modelo.
3. O orchestrator podia repetir uma conversão após ReadTimeout mesmo quando o Seed-VC continuava processando a tentativa original.

Após a primeira narração com **bernardo_full_v1**, o processo Seed-VC retinha aproximadamente 4,1 GiB de RAM anônima. A segunda narração tentou carregar **bernardo_gold_v1** sem remover o primeiro modelo. Os logs registraram três carregamentos concorrentes do mesmo checkpoint Gold.

O WSL estava limitado a aproximadamente 7,7 GiB de RAM e 2 GiB de swap. Antes da correção, havia somente cerca de 106 MiB disponíveis e o swap estava completamente ocupado. A CPU elevada e os acessos intermitentes de 100% ao SSD eram efeitos de paginação contínua, não processamento útil.

As correções foram implementadas e validadas com três narrações reais consecutivas. A sequência Full → Gold → Gold concluiu sem travamento, sem carregamentos duplicados e com apenas um modelo residente.

## Evidências coletadas antes da correção

| Métrica | Valor observado |
| --- | ---: |
| RAM física do host | 15,8 GiB |
| RAM disponível no Windows | 1,3 GiB |
| Limite de RAM do WSL | 7,7 GiB |
| MemAvailable no WSL | ~106 MiB |
| Swap do WSL | 2,0 GiB / 2,0 GiB usado |
| Seed-VC RSS | ~4.104 MiB |
| Seed-VC swap | ~109 MiB |
| Kokoro RSS | ~1.661 MiB |
| VRAM total | 12.288 MiB |
| VRAM usada | ~3.917 MiB |

A VRAM ainda tinha mais de 8 GiB livres. O gargalo era RAM do sistema/WSL.

Trecho decisivo dos logs:

~~~text
Carregando modelo para run_name=bernardo_full_v1...
Modelo carregado...
Conversão concluída...

Carregando modelo para run_name=bernardo_gold_v1...
Carregando modelo para run_name=bernardo_gold_v1...
Carregando modelo para run_name=bernardo_gold_v1...
~~~

As três mensagens de carga Gold ocorreram antes de o cache receber a primeira instância. Isso confirma a condição de corrida.

## Achados

### [Crítico] Cache ilimitado de modelos

O dicionário **_modelo_cache** crescia para cada **run_name** usado e nunca expulsava modelos. Em batch com múltiplas vozes, o consumo acumulava pesos, vocoder e modelos auxiliares.

**Correção:** cache LRU configurável, com default de um modelo residente. Antes de trocar de voz, o modelo anterior e seu conditioning são removidos; garbage collection, limpeza CUDA e **malloc_trim** devolvem memória livre.

### [Crítico] Carregamento e inferência sem serialização no Seed-VC

Endpoints síncronos do FastAPI rodam em thread pool. A sequência “se não está no cache, carregar” não era atômica. Duas ou mais chamadas podiam carregar o mesmo modelo simultaneamente.

**Correção:** um **threading.RLock** cobre carga, troca de modelo e inferência. Teste concorrente com quatro threads confirma exatamente uma chamada a **load_models**.

### [Alto] Retry inseguro após ReadTimeout

O retry do orchestrator incluía **requests.Timeout**. Um ReadTimeout não cancela a computação no servidor. A repetição podia iniciar outra inferência enquanto a original continuava ativa.

**Correção:** conversões Seed-VC não têm retry automático de ReadTimeout. Falhas de conexão antes do processamento continuam tratadas onde são seguras. Cada resposta também é validada contra a existência e o tamanho do WAV produzido.

### [Alto] BackgroundTasks sem backpressure apropriado para batch

Cada requisição criava uma tarefa que podia ficar bloqueada no lock global. Lotes grandes gerariam muitas tarefas/threads aguardando a GPU.

**Correção:** fila FIFO limitada com um único worker dedicado. Jobs excedentes recebem HTTP 503 quando a fila atinge o limite, em vez de pressionar indefinidamente o processo.

### [Alto] Ausência de limites de recursos nos containers

Todos os containers enxergavam o limite total do WSL. Uma regressão podia consumir toda a memória e congelar o host.

**Correção:** limites configuráveis de memória e CPU para Seed-VC, Kokoro e orchestrator. O default do Seed-VC é 5 GiB; uma falha futura fica contida no container em vez de tomar o sistema inteiro.

### [Médio] Retenção de arenas CPU/CUDA

Após cargas e inferências, caches do allocator CUDA e arenas glibc continuavam no RSS.

**Correção:** **gc.collect**, **torch.cuda.empty_cache** e **malloc_trim(0)** após carga, eviction, conversão e limpeza de conditioning. **MALLOC_ARENA_MAX=2** reduz fragmentação e retenção por múltiplas arenas.

### [Médio] Cópias desnecessárias de arrays

A inferência criava cópias extras entre NumPy e Torch.

**Correção:** uso de **torch.from_numpy** e arrays contíguos nos pontos seguros, além de **torch.inference_mode**.

### [Médio] Cache de referência sem invalidação automática

A chave anterior considerava somente run_name e caminho. Substituir o WAV no mesmo caminho mantinha conditioning antigo e a estrutura podia crescer.

**Correção:** chave inclui caminho resolvido, tamanho e **mtime_ns**. Fingerprints antigas do mesmo arquivo/modelo são removidas.

### [Médio] Histórico de jobs sem limite

O dicionário de jobs crescia indefinidamente durante batches longos.

**Correção:** retenção configurável, com default de 1.000 registros. Ao atingir o limite, jobs terminais antigos são podados; jobs pendentes e ativos são preservados.

### [Médio] Paralelismo CPU excessivo

O processo Seed-VC tinha 66 threads. Sob falta de memória, muitas threads aumentavam contenção e atividade de paginação.

**Correção:** limites para PyTorch, OpenMP, MKL, OpenBLAS, Numba e tokenizers. O default Seed-VC usa duas threads internas e limite de quatro CPUs no container.

### [Baixo] Observabilidade insuficiente

O health endpoint não mostrava memória ou cache.

**Correção:** health do Seed-VC agora informa modelos carregados, máximo do cache, RSS, swap e memória CUDA. Logs registram recursos após cada evento importante. O health do orchestrator informa profundidade/capacidade da fila, estado do worker e jobs retidos.

## Controles implementados

| Variável | Default | Finalidade |
| --- | ---: | --- |
| **SEEDVC_MAX_CACHED_MODELS** | 1 | Máximo de modelos Seed-VC residentes |
| **SEEDVC_CPU_THREADS** | 2 | Threads internas de CPU |
| **SEEDVC_INFERENCE_LOCK_TIMEOUT_S** | 900 | Espera máxima pelo slot de inferência |
| **SEEDVC_TIMEOUT_S** | 600 | Read timeout do cliente orchestrator |
| **MAX_QUEUED_JOBS** | 100 | Capacidade da fila de batch |
| **MAX_RETAINED_JOBS** | 1000 | Histórico máximo de jobs em memória |
| **MALLOC_ARENA_MAX** | 2 | Limite de arenas glibc |
| **SEEDVC_MEMORY_LIMIT** | 5g | Limite do container Seed-VC |
| **KOKORO_MEMORY_LIMIT** | 2500m | Limite do container Kokoro |
| **ORCHESTRATOR_MEMORY_LIMIT** | 768m | Limite do orchestrator |
| **SEEDVC_CPU_LIMIT** | 4.0 | CPUs máximas do Seed-VC |
| **KOKORO_CPU_LIMIT** | 4.0 | CPUs máximas do Kokoro |
| **ORCHESTRATOR_CPU_LIMIT** | 2.0 | CPUs máximas do orchestrator |

## Validação real após a correção

Foram enfileiradas duas narrações de uma vez:

1. **bernardo_full_v1**, draft.
2. **bernardo_gold_v1**, draft.

O segundo job permaneceu **pendente** enquanto o primeiro estava **rodando**, provando que a fila não iniciou duas inferências concorrentes.

### Resultado

| Execução | Modelo | Resultado | Tempo total | Seed-VC após conversão |
| --- | --- | --- | ---: | --- |
| 1 | bernardo_full_v1 | concluído | 120,0 s | RSS 1.578,8 MiB; swap 119,9 MiB |
| 2 | bernardo_gold_v1 | concluído | 16,1 s | RSS 1.403,1 MiB; swap 119,8 MiB |
| 3 | bernardo_gold_v1 reutilizado | concluído | 6,1 s | RSS 1.413,8 MiB; swap 119,1 MiB |

Durante a troca:

~~~text
Descarregando modelo LRU: bernardo_full_v1
CUDA após descarregar: 8,3 MiB alocados / 22,0 MiB reservados
Carregando modelo: bernardo_gold_v1
Modelo Gold carregado em 9,4 s
~~~

Estado final:

| Métrica | Valor |
| --- | ---: |
| Modelos Seed-VC residentes | 1 — bernardo_gold_v1 |
| Seed-VC RSS | ~1.404 MiB |
| Seed-VC VRAM alocada | ~1.207 MiB |
| Seed-VC VRAM reservada | ~1.308 MiB |
| MemAvailable do WSL | ~1,7 GiB |
| Jobs concluídos | 3/3 |

A terceira execução com o mesmo modelo não recarregou pesos e não apresentou crescimento acumulativo relevante de RSS.

## Testes automatizados

A suíte contém 16 testes e cobre:

- quatro threads solicitando o mesmo modelo e somente uma carga efetiva;
- eviction LRU na troca de modelo/precisão;
- remoção do conditioning associado ao modelo expulso;
- serialização de duas inferências simultâneas;
- liberação do lock mesmo se o cleanup falhar;
- ReadTimeout sem retry duplicado;
- poda de jobs terminais em batches longos;
- todos os testes anteriores de LUFS, True Peak, clipping, formato e determinismo.

Comando:

~~~bash
python -m unittest discover -s tests -v
~~~

## Riscos residuais e recomendações

### Swap global ainda ocupado

Após a validação, o swap do WSL permaneceu quase cheio porque páginas antigas não são trazidas de volta imediatamente. O WSL também executava vários containers de outros projetos, incluindo serviços de banco, frontend, workers e LiteLLM.

Isso não é um vazamento novo do Voice Workshop: o Seed-VC terminou com RSS estável e havia cerca de 1,7 GiB de memória disponível. Ainda assim, para operação de produção em batch:

- pare containers não relacionados quando não forem necessários;
- reinicie o WSL em uma janela de manutenção para zerar o swap histórico;
- não aumente indiscriminadamente o swap, pois isso troca um OOM controlado por SSD thrashing;
- monitore **MemAvailable**, não apenas **MemFree**;
- mantenha **SEEDVC_MAX_CACHED_MODELS=1** neste host.

### Arquivos finais

Os intermediários são removidos automaticamente, mas WAVs finais permanecem em **data/final_output**. Batches longos precisam de política externa de download, arquivamento e remoção. A auditoria não introduziu exclusão automática para evitar perda de resultados.

### Tamanho das narrações

A validação usou textos curtos de um chunk. Textos longos continuam serializados e protegidos por limites, mas exigem monitoramento de tempo e espaço em disco. O cache e a fila não crescem por chunk.

### Limites de container

O limite de 5 GiB do Seed-VC foi escolhido para conter regressões no host atual. Se modelos futuros legitimamente precisarem de mais memória, ajuste conscientemente junto com o limite do WSL e a memória deixada para Windows/Kokoro — nunca remova o limite sem telemetria.

## Arquivos alterados

- **seedvc/server.py**
- **orchestrator/run_pipeline.py**
- **orchestrator/api.py**
- **orchestrator/jobs.py**
- **docker-compose.yml**
- **.env.example**
- **tests/test_batch_stability.py**
- **BATCH_STABILITY_AUDIT.md**

## Conclusão

O defeito que inviabilizava batch foi corrigido em sua origem. O sistema agora possui:

- somente um modelo Seed-VC residente por padrão;
- carga e inferência mutuamente exclusivas;
- ausência de retry duplicado após timeout de leitura;
- fila limitada e serial;
- retenção limitada de jobs;
- limpeza explícita de memória;
- limites de CPU/RAM;
- telemetria operacional;
- testes de regressão;
- validação real Full → Gold → Gold.

A sequência crítica concluiu três vezes sem congelamento e sem crescimento acumulativo do modelo.
