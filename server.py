"""
server.py — läuft auf DEINEM Rechner. Serviert die Web-UI, lädt die Quelle nach R2,
erzeugt presigned URLs, startet den RunPod-Job, wartet und gibt das Ergebnis zurück.

Benötigte Umgebungsvariablen:
  RUNPOD_API_KEY        dein RunPod API-Key
  RUNPOD_ENDPOINT_ID    die Serverless-Endpoint-ID, aktuell: diirn17q2tgftf
  R2_ACCOUNT_ID         Cloudflare R2 Account-ID
  R2_ACCESS_KEY_ID      R2 Access Key
  R2_SECRET_ACCESS_KEY  R2 Secret
  R2_BUCKET             Bucket-Name

Starten:  uvicorn server:app --reload --port 8000
Dann im Browser: http://localhost:8000
"""

import os
import shutil
import subprocess
import tempfile
import time
import uuid

import boto3
import requests
from botocore.config import Config
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from starlette.background import BackgroundTask

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
LOCAL_FFPROBE = os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe") or "ffprobe"

VALID_TARGETS = {"original", "1080", "1440"}
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


def presign_get(key, ttl=3600):
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=ttl,
    )


def presign_put(key, ttl=3600):
    return s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": key, "ContentType": "video/mp4"},
        ExpiresIn=ttl,
    )


def _runpod_error_detail(status_payload):
    output = status_payload.get("output")
    if output:
        return output
    return status_payload


def _validate_local_video(path: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) < 2048:
        raise HTTPException(status_code=500, detail="Ergebnis fehlt oder ist verdächtig klein.")

    probe = subprocess.run(
        [
            LOCAL_FFPROBE,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height:format=duration",
            "-of", "default=noprint_wrappers=1",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Ergebnis ist keine gültige Videodatei: {probe.stderr[-1000:]}",
        )

    values = {}
    for line in probe.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    if values.get("codec_name") != "h264":
        raise HTTPException(
            status_code=500,
            detail=f"Ergebnis hat keinen H.264-Videostream: {probe.stdout}",
        )

    try:
        if float(values.get("duration", "0")) <= 0:
            raise ValueError
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail=f"Ergebnis hat keine gültige Dauer: {probe.stdout}",
        )


def _delete_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/process")
async def process(
    file: UploadFile,
    x: int = Form(...),
    y: int = Form(...),
    w: int = Form(...),
    h: int = Form(...),
    target: str = Form("original"),
):
    if target not in VALID_TARGETS:
        raise HTTPException(status_code=400, detail="target muss original, 1080 oder 1440 sein.")

    job_id = uuid.uuid4().hex
    src_key = f"in/{job_id}.mp4"
    out_key = f"out/{job_id}.mp4"

    # Quelle nach R2 hochladen
    s3.upload_fileobj(file.file, BUCKET, src_key, ExtraArgs={"ContentType": "video/mp4"})

    payload = {"input": {
        "video_url": presign_get(src_key),
        "upload_url": presign_put(out_key),
        "box": [x, y, w, h],
        "target": target,
    }}

    # Serverless-Job asynchron starten und pollen
    try:
        run_resp = requests.post(
            f"https://api.runpod.ai/v2/{ENDPOINT}/run",
            headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
            json=payload,
            timeout=60,
        )
        run_resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"RunPod-Start fehlgeschlagen: {exc}") from exc

    try:
        run = run_resp.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"RunPod returned non-JSON: {run_resp.text[:500]}",
        ) from exc

    if "id" not in run:
        raise HTTPException(status_code=502, detail=f"RunPod did not return job id: {run}")

    rid = run["id"]

    while True:
        try:
            status_resp = requests.get(
                f"https://api.runpod.ai/v2/{ENDPOINT}/status/{rid}",
                headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
                timeout=60,
            )
            status_resp.raise_for_status()
            st = status_resp.json()
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"RunPod-Status fehlgeschlagen: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="RunPod-Status war kein gültiges JSON.") from exc

        if st.get("status") in TERMINAL_STATUSES:
            break
        time.sleep(5)

    if st.get("status") != "COMPLETED":
        raise HTTPException(status_code=500, detail={"runpod_status": st.get("status"), "runpod_error": _runpod_error_detail(st)})

    output = st.get("output") or {}
    if isinstance(output, dict) and output.get("status") not in (None, "done"):
        raise HTTPException(status_code=500, detail={"runpod_status": "COMPLETED", "output": output})

    # Ergebnis aus R2 lokal herunterladen und validieren, bevor der Browser "fertig" sieht.
    tmp = os.path.join(tempfile.gettempdir(), f"clean_{job_id}.mp4")
    try:
        s3.download_file(BUCKET, out_key, tmp)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Konnte Ergebnis nicht aus R2 laden: {exc}") from exc

    _validate_local_video(tmp)

    return FileResponse(
        tmp,
        media_type="video/mp4",
        filename=f"clean_{job_id}.mp4",
        background=BackgroundTask(_delete_file, tmp),
    )
