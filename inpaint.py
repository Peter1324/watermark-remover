"""
inpaint.py
Watermark-Entfernung mit Crop-Trick + robustem H.264-Encoding.

- Inpainting läuft nur auf einem kleinen Fenster um das Wasserzeichen und wird in
  das Originalframe zurückgeklebt.
- Encoding läuft über ein explizites ffmpeg mit libx264, CRF 16, yuv420p und
  faststart, damit die MP4-Datei in Browsern/QuickTime sauber abspielbar ist.
- Output-Ziel: original, 1080 oder 1440. Es wird nie hochskaliert.
- Bei ffmpeg-/ffprobe-Fehlern wird hart abgebrochen, damit keine kaputte Datei
  als erfolgreicher Job hochgeladen wird.
"""

import math
import os
import subprocess
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
from simple_lama_inpainting import SimpleLama

FFMPEG = os.environ.get("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "/usr/bin/ffprobe")

# CRF steuert die Qualität: 16 = sehr hoch. Preset steuert Tempo/Dateigröße.
X264_PRESET = os.environ.get("X264_PRESET", "slow")
X264_CRF = os.environ.get("X264_CRF", "16")

# Wird einmal pro Worker geladen und bleibt im GPU-Speicher (warmer Worker -> schnell)
_lama = None


def get_model():
    global _lama
    if _lama is None:
        _lama = SimpleLama()  # lädt LaMa-Weights beim ersten Lauf (Apache-2.0)
    return _lama


def inpaint_region(frame_bgr: np.ndarray, box, pad: int = 24) -> np.ndarray:
    """
    frame_bgr: HxWx3 BGR numpy-Array (OpenCV-Frame)
    box: (x, y, w, h) Position des Wasserzeichens in Pixeln
    Rückgabe: BGR numpy-Array ohne Wasserzeichen.
    """
    H, W = frame_bgr.shape[:2]
    x, y, bw, bh = box

    # Fenster um das Wasserzeichen (auf Bildgrenzen begrenzt)
    wx0 = max(0, x - pad)
    wy0 = max(0, y - pad)
    wx1 = min(W, x + bw + pad)
    wy1 = min(H, y + bh + pad)

    window_bgr = frame_bgr[wy0:wy1, wx0:wx1]
    window_pil = Image.fromarray(cv2.cvtColor(window_bgr, cv2.COLOR_BGR2RGB))

    # Maske: weiß wo das Wasserzeichen ist, schwarz sonst (in Fenster-Koordinaten)
    mask = Image.new("L", window_pil.size, 0)
    d = ImageDraw.Draw(mask)
    mx0, my0 = x - wx0, y - wy0
    d.rectangle([mx0, my0, mx0 + bw, my0 + bh], fill=255)

    result_pil = get_model()(window_pil, mask).resize(window_pil.size)
    result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

    out = frame_bgr.copy()
    out[wy0:wy1, wx0:wx1] = result_bgr
    return out


def _run_probe(args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [FFPROBE, *args],
        capture_output=True,
        text=True,
    )


def _probe_fps(path: str) -> float:
    probe = _run_probe([
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ])
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe konnte FPS nicht lesen:\n{probe.stderr}")

    for line in probe.stdout.splitlines():
        line = line.strip()
        if not line or line == "0/0":
            continue
        if "/" in line:
            num, den = line.split("/", 1)
            den_f = float(den)
            if den_f != 0:
                fps = float(num) / den_f
                if fps > 0 and math.isfinite(fps):
                    return fps
        else:
            fps = float(line)
            if fps > 0 and math.isfinite(fps):
                return fps
    return 30.0


def _scale_filter(src_w: int, src_h: int, target: Optional[str]) -> Optional[str]:
    """target in {'original','1080','1440'}. Skaliert nie hoch."""
    if target in (None, "", "original"):
        return None

    aliases = {"1080p": "1080", "1440p": "1440", "2k": "1440", "2K": "1440"}
    target = aliases.get(str(target), str(target))
    if target not in {"1080", "1440"}:
        raise ValueError("target muss 'original', '1080' oder '1440' sein")

    target_h = int(target)
    if src_h <= target_h:
        return None  # nie hochskalieren
    return f"scale=-2:{target_h}"


def process_video(input_path: str, output_path: str, box, target: str = "original") -> None:
    """
    Streamt Frames über OpenCV, inpaintet pro Frame nur das Wasserzeichen-Fenster,
    pipet rohe Frames an ffmpeg und übernimmt Audio aus der Originaldatei.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Konnte Video nicht öffnen: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not math.isfinite(fps) or fps <= 0:
        fps = _probe_fps(input_path)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W <= 0 or H <= 0:
        cap.release()
        raise RuntimeError("Konnte Videoauflösung nicht lesen.")

    vf = _scale_filter(W, H, target)
    vf_args = ["-vf", vf] if vf else []

    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
        "-i", input_path,
        "-map", "0:v:0", "-map", "1:a:0?",
        *vf_args,
        "-c:v", "libx264", "-preset", X264_PRESET, "-crf", X264_CRF,
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]

    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = inpaint_region(frame, box)
            ff.stdin.write(np.ascontiguousarray(result, dtype=np.uint8).tobytes())
    except BrokenPipeError:
        # ffmpeg ist vorher gestorben; der konkrete Fehler wird unten aus stderr gelesen.
        pass
    finally:
        cap.release()
        try:
            if ff.stdin:
                ff.stdin.close()
        except BrokenPipeError:
            pass
        ff.wait()

    if ff.returncode != 0:
        err = ""
        if ff.stderr:
            err = ff.stderr.read().decode("utf-8", "ignore")[-4000:]
        raise RuntimeError(f"ffmpeg-Encoding fehlgeschlagen (code {ff.returncode}):\n{err}")

    _validate_output(output_path)


def _validate_output(path: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) < 2048:
        raise RuntimeError("Output-Datei fehlt oder ist verdächtig klein.")

    probe = _run_probe([
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height:format=duration",
        "-of", "default=noprint_wrappers=1",
        path,
    ])
    if probe.returncode != 0:
        raise RuntimeError(f"Output ist keine gültige Videodatei:\n{probe.stderr}")

    values = {}
    for line in probe.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    if values.get("codec_name") != "h264":
        raise RuntimeError(f"Output hat keinen H.264-Videostream:\n{probe.stdout}")

    duration = values.get("duration")
    try:
        if duration is None or float(duration) <= 0:
            raise ValueError
    except ValueError:
        raise RuntimeError(f"Output hat keine gültige Dauer:\n{probe.stdout}")
