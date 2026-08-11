"""
API HTTP fina sobre o pipeline Kokoro + Seed-VC.

Uso:
    uvicorn api:app --host 0.0.0.0 --port 9000
"""
import logging
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/narrar")
def narrar(req: NarracaoRequest, background_tasks: BackgroundTasks):
    if not req.texto.strip():
        raise HTTPException(status_code=400, detail="Campo 'texto' vazio")

    try:
        validar_run_name(req.run_name)
    except RunNameInvalido as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not _pipeline_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Já existe uma geração em andamento. Só uma GPU disponível -- tente novamente em instantes.",
        )

    try:
        resultado = gerar_narracao(req.texto, req.run_name)
    except Exception as e:
        logger.exception("Falha ao gerar narração")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        _pipeline_lock.release()

    # remove o .wav final do disco depois que a resposta terminar de ser
    # enviada -- não precisamos manter esses arquivos acumulando
    background_tasks.add_task(lambda p: Path(p).unlink(missing_ok=True), str(resultado))

    return FileResponse(
        path=str(resultado),
        media_type="audio/wav",
        filename=resultado.name,
        background=background_tasks,
    )