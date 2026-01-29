import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# ----------------------------
# Folders
# ----------------------------
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def safe_fps(cap, fallback=25.0) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        fps = float(fps)
    except Exception:
        fps = 0.0
    return fps if fps and fps > 1e-3 and not np.isnan(fps) else fallback


def even(n: int) -> int:
    # Some encoders want even width/height
    return n if n % 2 == 0 else n - 1


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="YOLO Video Detection", layout="wide")
st.title("🎥 YOLO Video Detection (No ffmpeg needed)")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", "best.pt")
    conf = st.slider("Confidence", 0.05, 1.0, 0.45, 0.01)
    preview_every = st.number_input("Preview every N frames (0 = off)", 0, 500, 0, 1)

uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

col_left, col_right = st.columns(2, gap="large")

if uploaded is None:
    st.info("Upload a video to start.")
    st.stop()

model_file = Path(model_path)
if not model_file.exists():
    st.error(f"Model file not found: {model_file.resolve()}")
    st.stop()

# Save upload to disk (stable for OpenCV + Streamlit)
in_path = TMP_DIR / f"upload_{uuid.uuid4().hex[:8]}_{Path(uploaded.name).name}"
with open(in_path, "wb") as f:
    f.write(uploaded.read())

with col_left:
    st.subheader("Original video")
    # Using bytes is often more reliable than passing a path
    st.video(in_path.read_bytes())

run = st.button("▶️ Run detection", type="primary", use_container_width=True)

if not run:
    st.stop()

# Load YOLO model
model = YOLO(str(model_file))

cap = cv2.VideoCapture(str(in_path))
if not cap.isOpened():
    st.error("Could not open the uploaded video.")
    st.stop()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
fps = safe_fps(cap)

w = even(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
h = even(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))

if w <= 0 or h <= 0:
    cap.release()
    st.error("Invalid video dimensions.")
    st.stop()

# ✅ AVI output with XVID codec (works without ffmpeg, high compatibility)
out_path = TMP_DIR / f"detected_{uuid.uuid4().hex[:8]}.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

progress = st.progress(0)
status = st.empty()
preview = st.empty()

processed = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Crop to even dims for codec safety
        frame = frame[:h, :w]

        results = model.predict(frame, conf=conf, verbose=False)
        annotated = results[0].plot()  # BGR with boxes

        writer.write(annotated)

        if preview_every and processed % int(preview_every) == 0:
            preview.image(annotated, channels="BGR", caption=f"Preview frame: {processed}")

        processed += 1
        if frame_count > 0:
            progress.progress(min(processed / frame_count, 1.0))
        status.write(f"Processed frames: {processed}" + (f" / {frame_count}" if frame_count else ""))

finally:
    cap.release()
    writer.release()
    progress.empty()
    status.empty()

# Validate output
if not out_path.exists() or out_path.stat().st_size == 0:
    st.error("Output video was not created (empty file).")
    st.stop()

with col_right:
    st.subheader("Detected video")
    st.caption(f"File: {out_path.name} | Size: {out_path.stat().st_size} bytes")
    # ✅ Most reliable playback: pass bytes
    st.video(out_path.read_bytes())

st.success("Done ✅")
