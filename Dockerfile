FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    portaudio19-dev \
    pulseaudio \
    libpulse-dev \
    ffmpeg \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "mavis.py"]