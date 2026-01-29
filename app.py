import os
import json
import time
import uuid

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
st.set_page_config(
    page_title="Fruit Quality AI", layout="wide", initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

        /* GLOBAL THEME */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
        }

        /* KEYFRAME ANIMATIONS */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse-red {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }

        @keyframes pulse-green {
            0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(34, 197, 94, 0); }
            100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
        }

        @keyframes border-flow {
            0% { border-color: #3b82f6; }
            50% { border-color: #8b5cf6; }
            100% { border-color: #3b82f6; }
        }

        /* CUSTOM CLASSES */
        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            animation: fadeIn 0.6s ease-out;
        }

        .hover-card {
            transition: all 0.3s ease;
        }
        .hover-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            border-color: #3b82f6;
        }

        .stat-value { font-size: 2rem; font-weight: 800; color: #f8fafc; }
        .stat-label { color: #94a3b8; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em; }

        /* QUALITY BADGES */
        .badge {
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.75rem;
            display: inline-block;
        }
        .badge-fresh {
            background: rgba(34, 197, 94, 0.2);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        .badge-rotten {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        /* LIVE FEED STYLING */
        .live-feed-container {
            border: 2px solid #334155;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }
        .recording-active {
            animation: border-flow 2s infinite;
            border-width: 2px;
        }
        .rec-dot {
            height: 12px; width: 12px;
            background-color: #ef4448;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            animation: pulse-red 2s infinite;
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background-color: #020617;
            border-right: 1px solid #1e293b;
        }
        
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
    try:
        if not os.path.exists(path):
            st.error(f"YOLO model file not found: {path}")
            st.stop()
        return YOLO(path)
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        st.stop()


def get_gemini_model():
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
    except Exception as e:
        st.warning(f"Gemini model initialization failed: {e}")
        return None


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
    try:
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
        (DETECTIONS_DIR / f"{det_id}.json").write_text(
            json.dumps(meta, ensure_ascii=False)
        )
        return det_id
    except Exception as e:
        st.warning(f"Failed to save detection: {e}")
        return ""


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
        except Exception as e:
            st.warning(f"Failed to load detection meta: {e}")
            continue
    return metas


def clear_storage():
    cleared = 0
    for folder in [TMP_DIR, DETECTIONS_DIR]:
        for f in folder.glob("*"):
            try:
                f.unlink()
                cleared += 1
            except Exception as e:
                st.warning(f"Failed to delete file {f}: {e}")
    st.session_state.active_det = None
    return cleared


# -----------------------------
# 3. SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown(
        "<h1 style='color: #38bdf8; font-weight: 800;'>🛡️ Fruit Guard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='margin-top: -15px; color: #94a3b8; font-size: 0.9em;'>AI-Powered Quality Control</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

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
    st.subheader("🎥 Intelligent Feed")

    input_source = st.radio(
        "Select Input Source", ["Upload Video", "Sample Video"], horizontal=True
    )

    video_file = None
    selected_sample_path = None

    if input_source == "Upload Video":
        video_file = st.file_uploader(
            "Upload Inspection Video", type=["mp4", "mov", "avi"]
        )
    else:
        sample_dir = Path("sample_videos")
        if sample_dir.exists():
            sample_files = sorted([f.name for f in sample_dir.glob("*.mp4")])
            selected_sample = st.selectbox("Choose a sample video", sample_files)
            if selected_sample:
                selected_sample_path = sample_dir / selected_sample
        else:
            st.warning("sample_videos directory not found.")

    video_screen = st.empty()

    b1, b2 = st.columns(2)

    # Enable start button if either a file is uploaded OR a sample is selected
    is_ready = (input_source == "Upload Video" and video_file is not None) or (
        input_source == "Sample Video" and selected_sample_path is not None
    )

    start_btn = b1.button(
        "🚀 Start Quality Analysis",
        use_container_width=True,
        type="primary",
        disabled=not is_ready,
    )
    stop_btn = b2.button("⏹ Stop", use_container_width=True)

    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False

    if stop_btn:
        st.session_state.stop_requested = True
        st.rerun()

    if start_btn and is_ready:
        st.session_state.stop_requested = False

        # Reset processed IDs for new run
        st.session_state.processed_ids = set()

        if not Path(model_path).exists():
            st.error(f"Model file not found: {model_path}")
            st.stop()

        final_video_path = None

        if input_source == "Upload Video":
            # Save uploaded video into tmp/
            tmp_video_path = (
                TMP_DIR / f"upload_{uuid.uuid4().hex[:8]}_{Path(video_file.name).name}"
            )
            try:
                with open(tmp_video_path, "wb") as f:
                    f.write(video_file.read())
                final_video_path = tmp_video_path
            except Exception as e:
                st.error(f"Failed to save uploaded video: {e}")
                st.stop()
        else:
            final_video_path = selected_sample_path

        try:
            cap = cv2.VideoCapture(str(final_video_path))
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

                try:
                    # Use YOLOv8 tracking
                    # persist=True is important for tracking objects across frames
                    results = model.track(
                        frame, conf=conf_threshold, persist=True, verbose=False
                    )
                except Exception as e:
                    st.warning(f"Prediction failed: {e}")
                    break

                now = time.time()
                # Save detections based on Unique Track ID
                for r in results:
                    for box in r.boxes:
                        try:
                            # Check if we have a track ID
                            if box.id is not None:
                                track_id = int(box.id.item())

                                # Only process if we haven't seen this ID before
                                if track_id not in st.session_state.processed_ids:
                                    cls_id = int(box.cls[0].detach().cpu().numpy())
                                    label = model.names.get(cls_id, str(cls_id))
                                    conf = _safe_float(box.conf[0])

                                    if conf >= save_only_above:
                                        det_id = save_detection(frame, box, label, conf)
                                        if det_id:
                                            st.session_state.processed_ids.add(track_id)
                                            last_save_time = now  # Keep for reference, though not primarily used for cooldown anymore
                        except Exception as e:
                            # Often happens if box.id is None during first few frames of detection or low confidence
                            pass

                try:
                    annotated_frame = results[0].plot()

                    # Styled container wrapper for the video feed
                    video_screen.empty()
                    with video_screen.container():
                        st.markdown(
                            """
                            <div class="live-feed-container recording-active">
                                <div style="position: absolute; top: 10px; left: 10px; z-index: 10; background: rgba(0,0,0,0.6); padding: 5px 10px; border-radius: 8px;">
                                    <span class="rec-dot"></span> <span style="color:white; font-weight:bold; font-size: 0.8em;">LIVE ANALYSIS</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.image(
                            annotated_frame, channels="BGR", use_container_width=True
                        )

                except Exception as e:
                    st.warning(f"Failed to display annotated frame: {e}")

            cap.release()
            st.success("Analysis Complete. Review findings in the right panel.")

        finally:
            # Cleanup only if it was an uploaded file
            if (
                input_source == "Upload Video"
                and final_video_path
                and final_video_path.exists()
            ):
                # Optional: delete tmp file
                pass


# -----------------------------
# 5. OBJECT GALLERY & AI CHAT
# -----------------------------
with col_ai:
    st.subheader("🔎 Inspection Gallery")

    # 1. DETAIL VIEW (TOP)
    active = st.session_state.active_det

    if active:
        # Styled Detail View
        st.markdown(
            f"""
        <div class="glass-card" style="border-left: 5px solid {"#ef4448" if "rotten" in active.get("label", "").lower() else "#22c55e"};">
            <h3 style="margin:0; padding-bottom: 10px;">Selected Inspection</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([1, 1.5], gap="medium")

        apath = active.get("image_path", "")
        alabel = active.get("label", "unknown_label")
        aconf = float(active.get("confidence", 0.0))

        with c1:
            if apath and Path(apath).exists():
                st.image(apath, use_container_width=True)
            else:
                st.warning("Active image missing")

        with c2:
            quality_class = (
                "badge-rotten" if "rotten" in alabel.lower() else "badge-fresh"
            )
            quality_text = "ROTTEN" if "rotten" in alabel.lower() else "FRESH"

            st.markdown(
                f"""
            <div style="font-size: 2em; font-weight: 800; margin-bottom: 5px;">{alabel.replace("_", " ").title()}</div>
            <div style="display:flex; gap: 10px; align-items:center; margin-bottom: 15px;">
                <span class="badge {quality_class}" style="font-size: 1em; padding: 6px 12px;">{quality_text}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"**Conf:** `{aconf:.1%}` | **Time:** `{active.get('timestamp', '--:--')}`"
            )

        # Gemini Chat for Active Item
        with st.expander("💬 Ask AI about this item", expanded=False):
            gemini = get_gemini_model()
            if not gemini:
                st.warning("Gemini API key not found.")
            else:
                chat_id = f"chat_{active.get('id', 'unknown')}"
                if chat_id not in st.session_state:
                    st.session_state[chat_id] = []

                for m in st.session_state[chat_id]:
                    with st.chat_message(m["role"]):
                        st.write(m["content"])

                prompt = st.chat_input("Ask question...", key="chat_input")
                if prompt:
                    st.session_state[chat_id].append(
                        {"role": "user", "content": prompt}
                    )
                    st.rerun()

                # Handle chat response after rerun if last message is user
                if (
                    st.session_state[chat_id]
                    and st.session_state[chat_id][-1]["role"] == "user"
                ):
                    with st.spinner("AI is thinking..."):
                        try:
                            last_prompt = st.session_state[chat_id][-1]["content"]
                            if apath and Path(apath).exists():
                                response = gemini.generate_content(
                                    [
                                        f"Context: YOLO detected this as {alabel}. Analysis asked: {last_prompt}",
                                        Image.open(apath).convert("RGB"),
                                    ]
                                )
                                answer = response.text or "No response."
                            else:
                                answer = "Image not found for AI analysis."
                        except Exception as e:
                            answer = f"Error: {e}"

                        st.session_state[chat_id].append(
                            {"role": "assistant", "content": answer}
                        )
                        st.rerun()

    else:
        st.info("Select an item from the list below to inspect details.", icon="👇")

    st.divider()

    # 2. SCROLLABLE LIST (BOTTOM)
    st.markdown("### Recent Detections")

    all_metas = load_detection_metas(limit=20)

    if not all_metas:
        st.caption("No detections yet.")
    else:
        # Scrollable container
        with st.container(height=500):
            # Use 2 columns for compact list
            cols = st.columns(2)

            for i, meta in enumerate(all_metas):
                with cols[i % 2]:
                    label = (meta.get("label") or "Unknown").replace("_", " ").title()
                    is_rotten = "rotten" in meta.get("label", "").lower()
                    border_color = "#ef4448" if is_rotten else "#22c55e"

                    with st.container(border=True):
                        c_img, c_info = st.columns([1, 1.5])

                        img_path = meta.get("image_path", "")
                        with c_img:
                            if img_path and Path(img_path).exists():
                                st.image(img_path, use_container_width=True)

                        with c_info:
                            st.markdown(f"**{label}**")
                            st.caption(f"{meta.get('timestamp')}")
                            if st.button(
                                "View",
                                key=f"btn_{meta['id']}",
                                use_container_width=True,
                            ):
                                st.session_state.active_det = meta
                                st.rerun()
