FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git ffmpeg && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Plachtaa/seed-vc.git /app
WORKDIR /app
RUN pip3 install -r requirements.txt

CMD ["tail", "-f", "/dev/null"]