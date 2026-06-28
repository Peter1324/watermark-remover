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


def _coerce_single_box(raw):
    """
    Normalisiert eine einzelne Box zu (x, y, w, h).
    Akzeptiert:
    - Dict: {"x": ..., "y": ..., "w": ..., "h": ...}
    - Liste/Tuple: [x, y, w, h]
    """
    if isinstance(raw, dict):
        if all(k in raw for k in ("x", "y", "w", "h")):
            x, y, w, h = raw["x"], raw["y"], raw["w"], raw["h"]
        else:
            raise ValueError("Box muss x, y, w, h enthalten.")
    elif isinstance(raw, (list, tuple)) and len(raw) == 4:
        x, y, w, h = raw
    else:
        raise ValueError("Ungültiges Box-Format.")

    try:
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
    except (TypeError, ValueError):
        raise ValueError("Box-Werte müssen Zahlen sein.")

    if not all(math.isfinite(v) for v in (x, y, w, h)):
        raise ValueError("Box-Werte müssen gültige Zahlen sein.")
    if w <= 0 or h <= 0:
        raise ValueError("Box-Breite und Box-Höhe müssen größer als 0 sein.")

    return (
        int(round(x)),
        int(round(y)),
        int(round(w)),
        int(round(h)),
    )


def _coerce_boxes(raw_boxes):
    """
    Macht aus einer einzelnen Box oder einer Box-Liste immer eine Liste von
    (x, y, w, h)-Tupeln. Dadurch bleibt der alte Single-Box-Flow kompatibel.
    """
    if raw_boxes is None:
        raise ValueError("Keine Boxen übergeben.")

    if isinstance(raw_boxes, dict):
        return [_coerce_single_box(raw_boxes)]

    if isinstance(raw_boxes, (list, tuple)):
        if len(raw_boxes) == 0:
            raise ValueError("Keine Boxen übergeben.")

        # Alte Form: [x, y, w, h]
        if len(raw_boxes) == 4 and not isinstance(raw_boxes[0], (dict, list, tuple)):
            return [_coerce_single_box(raw_boxes)]

        return [_coerce_single_box(b) for b in raw_boxes]

    raise ValueError("Ungültiges Boxen-Format.")


def _clamp_boxes(raw_boxes, W: int, H: int):
    """
    Schneidet Boxen auf die Bildgrenzen zu.
    Komplett außerhalb liegende Boxen werden verworfen.
    Wenn danach keine gültige Box übrig bleibt, wird hart abgebrochen.
    """
    boxes = _coerce_boxes(raw_boxes)
    clamped = []

    for x, y, w, h in boxes:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(W, x + w)
        y1 = min(H, y + h)

        if x1 <= x0 or y1 <= y0:
            continue

        clamped.append((x0, y0, x1 - x0, y1 - y0))

    if not clamped:
        raise ValueError("Keine gültige Markierung innerhalb des Videos gefunden.")

    return clamped


def _build_window_and_mask(boxes, W: int, H: int, pad: int = 24, mask_pad: int = 8):
    """
    Baut EIN gemeinsames Fenster um alle Boxen und EINE gemeinsame Maske.
    Diese Maske kann für alle Frames wiederverwendet werden, weil die Boxen
    aktuell statisch fürs ganze Video gelten.
    """
    wx0 = max(0, min(x for x, y, w, h in boxes) - pad)
    wy0 = max(0, min(y for x, y, w, h in boxes) - pad)
    wx1 = min(W, max(x + w for x, y, w, h in boxes) + pad)
    wy1 = min(H, max(y + h for x, y, w, h in boxes) + pad)

    if wx1 <= wx0 or wy1 <= wy0:
        raise ValueError("Markierungsfenster ist ungültig.")

    mask = Image.new("L", (wx1 - wx0, wy1 - wy0), 0)
    d = ImageDraw.Draw(mask)

    for x, y, w, h in boxes:
        mx0 = x - wx0
        my0 = y - wy0
        mx1 = mx0 + w
        my1 = my0 + h
        d.rectangle([mx0, my0, mx1, my1], fill=255)

    return (wx0, wy0, wx1, wy1), mask


def _inpaint_window(frame_bgr: np.ndarray, window, mask) -> np.ndarray:
    """
    Inpaintet genau ein Fenster mit genau einer gemeinsamen Maske.
    """
    wx0, wy0, wx1, wy1 = window

    window_bgr = frame_bgr[wy0:wy1, wx0:wx1]
    if window_bgr.size == 0:
        raise ValueError("Inpainting-Fenster ist leer.")

    window_pil = Image.fromarray(cv2.cvtColor(window_bgr, cv2.COLOR_BGR2RGB))
    result_pil = get_model()(window_pil, mask).resize(window_pil.size)
    result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

    out = frame_bgr.copy()
    out[wy0:wy1, wx0:wx1] = result_bgr
    return out


def inpaint_region(frame_bgr: np.ndarray, box, pad: int = 24) -> np.ndarray:
    """
    Rückwärtskompatibler Wrapper für den alten Single-Box-Flow.
    Intern nutzt er bereits die neue Multi-Box-Logik mit genau einer Box.
    """
    H, W = frame_bgr.shape[:2]
    boxes = _clamp_boxes(box, W, H)
    window, mask = _build_window_and_mask(boxes, W, H, pad=pad)
    return _inpaint_window(frame_bgr, window, mask)


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


def process_video(input_path: str, output_path: str, boxes, target: str = "original") -> None:
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

    boxes = _clamp_boxes(boxes, W, H)
    window, mask = _build_window_and_mask(boxes, W, H)

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
            result = _inpaint_window(frame, window, mask)
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
