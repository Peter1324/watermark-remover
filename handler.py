"""
handler.py — RunPod Serverless Einstiegspunkt.

Eingabe-JSON:
{
  "video_url":  "<presigned GET-URL zum Download der Quelle>",
  "upload_url": "<presigned PUT-URL zum Hochladen des Ergebnisses>",
  "box": [x, y, w, h],
  "target": "original" | "1080" | "1440"
}

Bei Verarbeitungsfehlern wird eine Exception geworfen. Dadurch markiert RunPod
den Job als FAILED und es wird keine kaputte Datei hochgeladen.
"""

import os
import tempfile
import requests
import runpod
from inpaint import process_video


def _download(url: str, path: str):
    with requests.get(url, stream=True, timeout=900) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):  # 1 MB Chunks
                if chunk:
                    f.write(chunk)


def _upload(url: str, path: str):
    with open(path, "rb") as f:
        r = requests.put(
            url,
            data=f,
            headers={"Content-Type": "video/mp4"},
            timeout=900,
        )
        r.raise_for_status()


def handler(job):
    job_input = job["input"]
    box = tuple(job_input["box"])
    target = job_input.get("target", "original")

    with tempfile.TemporaryDirectory() as work:
        src = os.path.join(work, "src.mp4")
        out = os.path.join(work, "out.mp4")

        try:
            _download(job_input["video_url"], src)
            process_video(src, out, box, target=target)
            _upload(job_input["upload_url"], out)
        except Exception as exc:
            # Wichtig: nicht schlucken und nicht uploaden. RunPod soll FAILED melden.
            raise RuntimeError(f"Verarbeitung fehlgeschlagen: {exc}") from exc

    return {"status": "done", "target": target}


runpod.serverless.start({"handler": handler})
