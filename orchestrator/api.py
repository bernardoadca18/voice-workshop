"""
API HTTP fina sobre o pipeline Kokoro + Seed-VC.

Uso:
    uvicorn api:app --host 0.0.0.0 --port 9000
"""
import logging
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from jobs import Job, JobStatus, criar_job, obter_job, atualizar_job, remover_job
from run_pipeline import gerar_narracao, validar_run_name, RunNameInvalido

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="Voice Workshop API", version="0.1.0")

# Só existe uma GPU -- as duas etapas do pipeline (Kokoro/Seed-VC) não
# podem processar duas requisições em paralelo sem disputar VRAM e
# colidir em diretórios/containers compartilhados. Serializa aqui.
_pipeline_lock = threading.Lock()


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/narrar")
def narrar(req: NarracaoRequest, background_tasks: BackgroundTasks):
    """
    Inicia uma narração de forma assíncrona.
    Retorna imediatamente com um job_id para consultar o status.
    """
    if not req.texto.strip():
        raise HTTPException(status_code=400, detail="Campo 'texto' vazio")

    try:
        validar_run_name(req.run_name)
    except RunNameInvalido as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    job = criar_job()
    background_tasks.add_task(_processar_job, job.id, req.texto, req.run_name, req.qualidade)

    return {"job_id": job.id, "status_url": f"/jobs/{job.id}"}


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
