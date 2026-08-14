"""
Job store em memória para processamento assíncrono de narrações.

NOTA IMPORTANTE: Esta implementação é intencionalmente simples — um dict
em memória protegido por lock. Não há persistência em disco nem Redis.
Isso é adequado para um orchestrator de réplica única que raramente reinicia.

Migração futura para Redis/SQLite seria necessária se:
- O orchestrator passar a rodar em múltiplas réplicas
- Reinícios frequentes causarem perda de jobs em andamento
- For necessário consultar status de jobs de fora do processo

Por enquanto, manter simples é o certo — não otimizar prematuramente.
"""
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDENTE = "pendente"
    RODANDO = "rodando"
    CONCLUIDO = "concluido"
    ERRO = "erro"


class Job(BaseModel):
    id: str
    status: JobStatus
    chunk_atual: int = 0
    total_chunks: int = 0
    resultado_path: Optional[str] = None
    erro: Optional[str] = None
    criado_em: datetime


# Job store em memória — dict protegido por lock
_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def criar_job() -> Job:
    """Cria um novo job com status PENDENTE e retorna-o."""
    job_id = uuid.uuid4().hex
    job = Job(
        id=job_id,
        status=JobStatus.PENDENTE,
        criado_em=datetime.now(timezone.utc),
    )
    with _jobs_lock:
        _jobs[job_id] = job
    return job


def obter_job(job_id: str) -> Optional[Job]:
    """Obtém um job pelo ID, ou None se não existir."""
    with _jobs_lock:
        return _jobs.get(job_id)


def atualizar_job(job_id: str, **kwargs) -> None:
    """Atualiza campos de um job existente."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)


def remover_job(job_id: str) -> None:
    """Remove um job do store."""
    with _jobs_lock:
        _jobs.pop(job_id, None)
