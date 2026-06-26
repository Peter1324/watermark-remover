# CUDA + Python Basis. Aktuellen Tag prüfen: https://hub.docker.com/r/runpod/base
FROM runpod/base:0.6.3-cuda12.2.0

# ffmpeg/ffprobe + OpenCV-Systemabhängigkeiten
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (Optional, aber empfohlen) LaMa-Weights ins Image backen -> killt Cold-Start-Download:
# RUN python -c "from simple_lama_inpainting import SimpleLama; SimpleLama()"

COPY inpaint.py handler.py ./
CMD ["python", "-u", "handler.py"]
