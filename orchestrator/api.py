"""
API HTTP fina sobre o pipeline Kokoro + Seed-VC.

Uso:
    uvicorn api:app --host 0.0.0.0 --port 9000
"""
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from jobs import Job, JobStatus, criar_job, obter_job, atualizar_job, remover_job, quantidade_jobs
from run_pipeline import gerar_narracao, validar_run_name, RunNameInvalido

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="Voice Workshop API", version="0.1.0")

# Só existe uma GPU -- as duas etapas do pipeline (Kokoro/Seed-VC) não
# podem processar duas requisições em paralelo sem disputar VRAM e
# colidir em diretórios/containers compartilhados. Serializa aqui.
_pipeline_lock = threading.Lock()
MAX_QUEUED_JOBS = max(1, int(os.environ.get("MAX_QUEUED_JOBS", "100")))
_job_queue: queue.Queue[tuple[str, str, str, str]] = queue.Queue(
    maxsize=MAX_QUEUED_JOBS
)


class NarracaoRequest(BaseModel):
    texto: str
    run_name: str = "minha_voz_v1"
    qualidade: Literal["draft", "final"] = "final"


def _processar_job(job_id: str, texto: str, run_name: str, qualidade: str) -> None:
    """Processa uma narração em background, atualizando o job status."""
    atualizar_job(job_id, status=JobStatus.RODANDO)

    try:
        with _pipeline_lock:
            resultado = gerar_narracao(texto, run_name, qualidade=qualidade)
        atualizar_job(job_id, status=JobStatus.CONCLUIDO, resultado_path=str(resultado))
        logger.info("Job %s concluído com sucesso", job_id)
    except Exception as e:
        logger.exception("Falha no job %s", job_id)
        atualizar_job(job_id, status=JobStatus.ERRO, erro=str(e))


def _job_worker() -> None:
    """Único consumidor: batch vira fila, nunca uma pilha de threads bloqueadas."""
    while True:
        payload = _job_queue.get()
        try:
            _processar_job(*payload)
        finally:
            _job_queue.task_done()


_worker_thread = threading.Thread(
    target=_job_worker,
    name="voice-workshop-job-worker",
    daemon=True,
)
_worker_thread.start()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "queue_depth": _job_queue.qsize(),
        "queue_capacity": MAX_QUEUED_JOBS,
        "worker_alive": _worker_thread.is_alive(),
        "retained_jobs": quantidade_jobs(),
    }


@app.post("/narrar")
def narrar(req: NarracaoRequest):
    """Enfileira a narração sem criar uma thread bloqueada por requisição."""
    if not req.texto.strip():
        raise HTTPException(status_code=400, detail="Campo 'texto' vazio")

    try:
        validar_run_name(req.run_name)
    except RunNameInvalido as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    job = criar_job()
    try:
        _job_queue.put_nowait((job.id, req.texto, req.run_name, req.qualidade))
    except queue.Full as exc:
        remover_job(job.id)
        raise HTTPException(
            status_code=503,
            detail=f"Fila de narração cheia (limite: {MAX_QUEUED_JOBS})",
        ) from exc

    return {
        "job_id": job.id,
        "status_url": f"/jobs/{job.id}",
        "queue_depth": _job_queue.qsize(),
    }


@app.get("/jobs/{job_id}")
def status_job(job_id: str):
    """Consulta o status de um job."""
    job = obter_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return job


@app.get("/jobs/{job_id}/audio")
def baixar_audio(job_id: str):
    """Baixa o áudio de um job concluído."""
    job = obter_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    if job.status != JobStatus.CONCLUIDO:
        raise HTTPException(status_code=404, detail="Áudio ainda não disponível")
    if job.resultado_path is None:
        raise HTTPException(status_code=404, detail="Áudio não disponível")

    if not Path(job.resultado_path).exists():
        raise HTTPException(status_code=404, detail="Arquivo de áudio não encontrado")

    return FileResponse(
        path=job.resultado_path,
        media_type="audio/wav",
        filename=Path(job.resultado_path).name,
    )
