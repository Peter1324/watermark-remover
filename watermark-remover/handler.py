"""
handler.py — RunPod Serverless Einstiegspunkt.

Eingabe-JSON:
{
  "video_url":  "<presigned GET-URL zum Download der Quelle>",
  "upload_url": "<presigned PUT-URL zum Hochladen des Ergebnisses>",
  "box": [x, y, w, h]
}
Rückgabe: {"status": "done"}  (das Ergebnis landet per upload_url in deinem Bucket)
"""

import os
import tempfile
import requests
import runpod
from inpaint import process_video


def _download(url: str, path: str):
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):  # 1 MB Chunks
                f.write(chunk)


def _upload(url: str, path: str):
    with open(path, "rb") as f:
        r = requests.put(url, data=f,
                         headers={"Content-Type": "video/mp4"}, timeout=600)
        r.raise_for_status()


def handler(job):
    job_input = job["input"]
    box = tuple(job_input["box"])

    work = tempfile.mkdtemp()
    src = os.path.join(work, "src.mp4")
    out = os.path.join(work, "out.mp4")

    _download(job_input["video_url"], src)
    process_video(src, out, box)
    _upload(job_input["upload_url"], out)

    return {"status": "done"}


runpod.serverless.start({"handler": handler})
