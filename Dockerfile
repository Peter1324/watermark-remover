FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# --- ffmpeg mit libx264 (Fix für "Unknown encoder 'libx264'") ---
# Wichtig: PyTorch/Conda-Images bringen oft ein eigenes ffmpeg ohne GPL/x264 mit.
# Deshalb installieren wir das System-ffmpeg, entfernen conda-ffmpeg und nutzen danach
# bewusst absolute Pfade über FFMPEG_BIN/FFPROBE_BIN.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

RUN rm -f /opt/conda/bin/ffmpeg /opt/conda/bin/ffprobe 2>/dev/null || true

RUN /usr/bin/ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 \
    || (echo "BUILD-ABBRUCH: libx264 fehlt im ffmpeg!" && exit 1)

ENV FFMPEG_BIN=/usr/bin/ffmpeg
ENV FFPROBE_BIN=/usr/bin/ffprobe
ENV X264_PRESET=slow
ENV X264_CRF=16

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY inpaint.py handler.py ./

CMD ["python", "-u", "handler.py"]
