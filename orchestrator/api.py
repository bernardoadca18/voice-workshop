"""
API HTTP fina sobre o pipeline Kokoro + Seed-VC.

Uso:
    uvicorn api:app --host 0.0.0.0 --port 9000

Endpoint principal:
    POST /narrar {"texto": "...", "run_name": "minha_voz_v1"}
    -> retorna o arquivo .wav final gerado
"""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from run_pipeline import gerar_narracao

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="Voice Workshop API", version="0.1.0")


class NarracaoRequest(BaseModel):
    texto: str
    run_name: str = "minha_voz_v1"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/narrar")
def narrar(req: NarracaoRequest):
    if not req.texto.strip():
        raise HTTPException(status_code=400, detail="Campo 'texto' vazio")

    try:
        resultado = gerar_narracao(req.texto, req.run_name)
    except Exception as e:
        logger.exception("Falha ao gerar narração")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return FileResponse(
        path=str(resultado),
        media_type="audio/wav",
        filename=resultado.name,
    )