# Watermark Remover

Fertiger Projektordner aus dem Blueprint.

## Lokal starten

```bash
pip install fastapi uvicorn boto3 requests python-multipart

export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...
export R2_ACCOUNT_ID=...
export R2_ACCESS_KEY_ID=...
export R2_SECRET_ACCESS_KEY=...
export R2_BUCKET=watermark-jobs

uvicorn server:app --port 8000
```

Dann öffnen: http://localhost:8000

## Dateien

- `inpaint.py` — Inpainting-Kern
- `handler.py` — RunPod Serverless Handler
- `requirements.txt` — GPU-Container Dependencies
- `Dockerfile` — RunPod Docker Image
- `server.py` — lokale FastAPI Steuer-App
- `index.html` — Browser UI
