import os
import json
import time
import uuid
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import google.generativeai as genai


# -----------------------------
# 1. UI CONFIG & STYLING
# -----------------------------
st.set_page_config(page_title="Fruit Quality AI", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .detection-card {
        padding: 12px; border-radius: 12px; background: #1c2128;
        border: 1px solid #30363d; margin-bottom: 15px;
    }
    .quality-fresh { color: #3fb950; font-weight: bold; }
    .quality-rotten { color: #f85149; font-weight: bold; }
    .gallery-item { cursor: pointer; border: 2px solid transparent; transition: 0.3s; }
    .gallery-item:hover { border-color: #58a6ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

DETECTIONS_DIR = Path("saved_detections")
DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Session defaults
if "active_det" not in st.session_state:
    st.session_state.active_det = None


# -----------------------------
# 2. MODELS & UTILS
# -----------------------------
@st.cache_resource
def load_yolo_model(path: str):
    return YOLO(path)


def get_gemini_model():
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        try:
            return float(x.item())
        except Exception:
            return 0.0


def save_detection(frame_bgr: np.ndarray, box, label: str, conf) -> str:
    """
    Saves a detection crop + JSON metadata.
    """
    det_id = str(uuid.uuid4())[:8]

    xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
    x1, y1, x2, y2 = xyxy

    # Padding for better context
    h, w = frame_bgr.shape[:2]
    pad = 20
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad)
    y2p = min(h, y2 + pad)

    crop = frame_bgr[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return ""

    img_path = DETECTIONS_DIR / f"{det_id}.jpg"
    cv2.imwrite(str(img_path), crop)

    meta = {
        "id": det_id,
        "label": str(label),
        "confidence": _safe_float(conf),
        "timestamp": time.strftime("%H:%M:%S"),
        "image_path": str(img_path),
    }
    (DETECTIONS_DIR / f"{det_id}.json").write_text(json.dumps(meta, ensure_ascii=False))
    return det_id


def load_detection_metas(limit: int = 12):
    """
    Loads detection JSON files safely.
    Backward compatible: if 'class_name' exists but 'label' doesn't, it is mapped.
    """
    files = sorted(DETECTIONS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    metas = []
    for f in files[: max(limit * 3, limit)]:
        try:
            meta = json.loads(f.read_text())
            if "label" not in meta and "class_name" in meta:
                meta["label"] = meta["class_name"]
            if "id" in meta and "image_path" in meta and "label" in meta:
                metas.append(meta)
            if len(metas) >= limit:
                break
        except Exception:
            continue
    return metas


def clear_storage():
    cleared = 0
    for folder in [TMP_DIR, DETECTIONS_DIR]:
        for f in folder.glob("*"):
            try:
                f.unlink()
                cleared += 1
            except Exception:
                pass
    st.session_state.active_det = None
    return cleared


# -----------------------------
# 3. SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("🛡️ Fruit Quality Guard")

    model_path = st.text_input("Model Path", "best.pt")
    conf_threshold = st.slider("Min Confidence (display)", 0.1, 1.0, 0.45)

    st.divider()
    save_only_above = st.slider("Save detections only above", 0.1, 1.0, 0.80)
    cooldown_sec = st.slider("Cooldown between saves (sec)", 0.0, 10.0, 2.0, 0.5)

    st.divider()
    if st.button("🧹 Clear All Storage (tmp + detections)", use_container_width=True):
        n = clear_storage()
        st.success(f"Cleared {n} stored files.")
        st.rerun()

    st.caption(
        "Expected classes: fresh_apple, fresh_banana, fresh_orange, "
        "rotten_apple, rotten_banana, rotten_orange."
    )


# -----------------------------
# 4. MAIN WORKSPACE
# -----------------------------
col_vid, col_ai = st.columns([1.4, 1], gap="large")

with col_vid:
    st.subheader("🎥 Intelligent Feed (Uploaded Video)")
    video_file = st.file_uploader("Upload Inspection Video", type=["mp4", "mov", "avi"])
    video_screen = st.empty()

    b1, b2 = st.columns(2)
    start_btn = b1.button("🚀 Start Quality Analysis", use_container_width=True, type="primary", disabled=(video_file is None))
    stop_btn = b2.button("⏹ Stop", use_container_width=True)

    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False

    if stop_btn:
        st.session_state.stop_requested = True
        st.warning("Stop requested. The app will stop on the next rerun.")
        st.rerun()

    if start_btn and video_file:
        st.session_state.stop_requested = False

        if not Path(model_path).exists():
            st.error(f"Model file not found: {model_path}")
            st.stop()

        # Save uploaded video into tmp/
        tmp_video_path = TMP_DIR / f"upload_{uuid.uuid4().hex[:8]}_{Path(video_file.name).name}"
        with open(tmp_video_path, "wb") as f:
            f.write(video_file.read())

        try:
            cap = cv2.VideoCapture(str(tmp_video_path))
            if not cap.isOpened():
                st.error("Could not open the uploaded video.")
                st.stop()

            model = load_yolo_model(model_path)

            last_save_time = 0.0

            while cap.isOpened():
                if st.session_state.stop_requested:
                    st.warning("Stopped by user.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=conf_threshold, verbose=False)

                now = time.time()
                # Save detections with cooldown (prevents duplicates)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].detach().cpu().numpy())
                        label = model.names.get(cls_id, str(cls_id))
                        conf = _safe_float(box.conf[0])

                        if conf >= save_only_above and (now - last_save_time) >= cooldown_sec:
                            det_id = save_detection(frame, box, label, conf)
                            if det_id:
                                last_save_time = now

                annotated_frame = results[0].plot()
                video_screen.image(annotated_frame, channels="BGR", use_container_width=True)

            cap.release()
            st.success("Analysis Complete. Review findings in the right panel.")

        finally:
            # Keep tmp video unless you want to auto-delete:
            # try: tmp_video_path.unlink()
            # except Exception: pass
            pass


# -----------------------------
# 5. OBJECT GALLERY & AI CHAT
# -----------------------------
with col_ai:
    st.subheader("🔎 Inspection Gallery")

    all_metas = load_detection_metas(limit=12)

    if not all_metas:
        st.info("No detections saved yet.")
    else:
        cols = st.columns(3)

        for i, meta in enumerate(all_metas):
            with cols[i % 3]:
                label_text = (meta.get("label") or "").lower()
                quality_style = "quality-rotten" if "rotten" in label_text else "quality-fresh"

                img_path = meta.get("image_path", "")
                if img_path and Path(img_path).exists():
                    st.image(img_path, use_container_width=True)
                else:
                    st.warning("Missing image file")

                st.markdown(
                    f"<div class='{quality_style}'>{meta.get('label','Unknown')}</div>",
                    unsafe_allow_html=True,
                )

                if st.button(f"Inspect ID: {meta.get('id','????')}", key=f"inspect_{meta.get('id','x')}"):
                    st.session_state.active_det = meta
                    st.rerun()

        st.divider()

        active = st.session_state.active_det
        if active:
            with st.container(border=True):
                c1, c2 = st.columns([1, 2], gap="large")

                apath = active.get("image_path", "")
                alabel = active.get("label", "unknown_label")
                aconf = float(active.get("confidence", 0.0))

                if apath and Path(apath).exists():
                    c1.image(apath, use_container_width=True)
                else:
                    c1.warning("Active image missing")

                quality_class = "🔴 ROTTEN" if "rotten" in alabel.lower() else "🟢 FRESH"
                c2.markdown(f"### {alabel.replace('_', ' ').title()}")
                c2.markdown(f"**Quality Status:** {quality_class}")
                c2.write(f"Confidence: {aconf:.2%}")

            gemini = get_gemini_model()
            if not gemini:
                st.warning("Gemini API key not found. Add GEMINI_API_KEY to secrets or environment variables.")
            else:
                st.markdown("#### 💬 Ask Gemini about this item")

                chat_id = f"chat_{active.get('id','unknown')}"
                if chat_id not in st.session_state:
                    st.session_state[chat_id] = []

                for m in st.session_state[chat_id]:
                    with st.chat_message(m["role"]):
                        st.write(m["content"])

                prompt = st.chat_input("Ex: Why is this considered rotten? Any handling advice?")
                if prompt:
                    st.session_state[chat_id].append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.write(prompt)

                    with st.chat_message("assistant"):
                        try:
                            if apath and Path(apath).exists():
                                response = gemini.generate_content(
                                    [
                                        f"Context: YOLO detected this as {alabel}. "
                                        "Analyse visible signs of bruising, mould, discoloration, or ripeness. "
                                        "Be practical and state uncertainty if needed.",
                                        Image.open(apath).convert("RGB"),
                                        prompt,
                                    ]
                                )
                                answer = response.text or ""
                            else:
                                answer = "I cannot open the saved crop image. Please rerun detection."
                        except Exception as e:
                            answer = f"Gemini error: {e}"

                        st.write(answer)
                        st.session_state[chat_id].append({"role": "assistant", "content": answer})
