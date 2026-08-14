FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git ffmpeg curl && rm -rf /var/lib/apt/lists/*

# Clone do repositório Seed-VC upstream
# NOTA: Para reprodutibilidade total, descomente a linha do checkout com o hash específico:
# RUN git clone https://github.com/Plachtaa/seed-vc.git /app && cd /app && git checkout <commit-hash>
RUN git clone https://github.com/Plachtaa/seed-vc.git /app
WORKDIR /app
RUN pip3 install -r requirements.txt fastapi "uvicorn[standard]"

COPY server.py /app/server.py

# Healthcheck: verifica se o processo está vivo (não garante modelo carregado)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]