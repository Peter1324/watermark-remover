"""
inpaint.py
Kern der Watermark-Entfernung.

Kernidee: Das Wasserzeichen ist klein und immer an derselben Stelle. Wir lassen das
Modell deshalb nur auf einem kleinen Fenster um das Wasserzeichen laufen (plus Rand
als Kontext) und kleben das Ergebnis zurück ins volle Bild. Das macht 4K fast so
billig wie 1080p und vermeidet Schmieren im Rest des Bildes.
"""

import os
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw
from simple_lama_inpainting import SimpleLama

# Wird einmal pro Worker geladen und bleibt im GPU-Speicher (warmer Worker -> schnell)
_lama = None
def get_model():
    global _lama
    if _lama is None:
        _lama = SimpleLama()   # lädt LaMa-Weights beim ersten Lauf (Apache-2.0)
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

    # PIL/LaMa arbeitet in RGB
    window_rgb = cv2.cvtColor(window_bgr, cv2.COLOR_BGR2RGB)
    window_pil = Image.fromarray(window_rgb)

    # Maske: weiß wo das Wasserzeichen ist, schwarz sonst (in Fenster-Koordinaten)
    mask = Image.new("L", window_pil.size, 0)
    d = ImageDraw.Draw(mask)
    mx0, my0 = x - wx0, y - wy0
    d.rectangle([mx0, my0, mx0 + bw, my0 + bh], fill=255)

    result_pil = get_model()(window_pil, mask)          # PIL RGB raus
    result_pil = result_pil.resize(window_pil.size)     # Sicherheit (Größe angleichen)
    result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

    out = frame_bgr.copy()
    out[wy0:wy1, wx0:wx1] = result_bgr
    return out


def _probe_fps(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "0", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True
    ).stdout.strip()
    num, den = out.split("/")
    return float(num) / float(den)


def process_video(input_path: str, output_path: str, box) -> None:
    """
    Streamt Frames über OpenCV (wenig Speicherplatzbedarf), inpaintet das Fenster
    pro Frame, pipet rohe Frames an ffmpeg zur H.264-Kodierung und mischt am Ende
    den Originalton wieder rein.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Konnte Video nicht öffnen: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or _probe_fps(input_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    silent = output_path + ".silent.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", f"{fps}",
         "-i", "-",
         "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", 
         silent],
        stdin=subprocess.PIPE
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = inpaint_region(frame, box)
            ff.stdin.write(result.astype(np.uint8).tobytes())
    finally:
        cap.release()
        ff.stdin.close()
        ff.wait()

    # Originalton wieder reinmischen (fällt still weg, wenn die Quelle keinen Ton hat)
    subprocess.run(
        ["ffmpeg", "-y",
         "-i", silent, "-i", input_path,
         "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0?",
         "-c:a", "aac", "-shortest", output_path],
        check=True
    )
    os.remove(silent)
