"""
server.py — läuft auf DEINEM Rechner. Serviert die Web-UI, lädt die Quelle nach R2,
erzeugt presigned URLs, startet den RunPod-Job, wartet und gibt das Ergebnis zurück.

Benötigte Umgebungsvariablen:
  RUNPOD_API_KEY        dein RunPod API-Key
  RUNPOD_ENDPOINT_ID    die Serverless-Endpoint-ID
  R2_ACCOUNT_ID         Cloudflare R2 Account-ID
  R2_ACCESS_KEY_ID      R2 Access Key
  R2_SECRET_ACCESS_KEY  R2 Secret
  R2_BUCKET             Bucket-Name

Starten:  uvicorn server:app --reload --port 8000
Dann im Browser: http://localhost:8000
"""

import os, uuid, time
import boto3
import requests
from botocore.config import Config
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI()

# --- R2 (S3-kompatibel) Client ---
s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    config=Config(signature_version="s3v4"),
)
BUCKET = os.environ["R2_BUCKET"]
RUNPOD_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT = os.environ["RUNPOD_ENDPOINT_ID"]


def presign_get(key, ttl=3600):
    return s3.generate_presigned_url("get_object",
        Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=ttl)

def presign_put(key, ttl=3600):
    return s3.generate_presigned_url("put_object",
        Params={"Bucket": BUCKET, "Key": key, "ContentType": "video/mp4"},
        ExpiresIn=ttl)


@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/process")
async def process(file: UploadFile,
                  x: int = Form(...), y: int = Form(...),
                  w: int = Form(...), h: int = Form(...)):
    job_id = uuid.uuid4().hex
    src_key = f"in/{job_id}.mp4"
    out_key = f"out/{job_id}.mp4"

    # Quelle nach R2 hochladen
    s3.upload_fileobj(file.file, BUCKET, src_key,
                      ExtraArgs={"ContentType": "video/mp4"})

    payload = {"input": {
        "video_url": presign_get(src_key),
        "upload_url": presign_put(out_key),
        "box": [x, y, w, h],
    }}

    # Serverless-Job asynchron starten und pollen
    run = requests.post(
        f"https://api.runpod.ai/v2/{ENDPOINT}/run",
        headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
        json=payload, timeout=60,
    ).json()
    rid = run["id"]

    while True:
        st = requests.get(
            f"https://api.runpod.ai/v2/{ENDPOINT}/status/{rid}",
            headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
            timeout=60,
        ).json()
        if st.get("status") in ("COMPLETED", "FAILED"):
            break
        time.sleep(5)

    if st.get("status") != "COMPLETED":
        return {"error": st}

    # fertiges File aus R2 zurück an den Browser streamen
    obj = s3.get_object(Bucket=BUCKET, Key=out_key)
    return StreamingResponse(obj["Body"],
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="clean_{job_id}.mp4"'})
