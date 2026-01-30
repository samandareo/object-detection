import os
import json
import time
import uuid

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import requests
from PIL import Image
from ultralytics import YOLO
import google.generativeai as genai
from streamlit_lottie import st_lottie

# -----------------------------
# 1. UI CONFIG & STYLING
# -----------------------------
st.set_page_config(
    page_title="Fruit Guard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom Lottie Loader
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Load Assets
# Robot scanning animation
LOTTIE_ROBOT = "https://assets9.lottiefiles.com/packages/lf20_aa0wy04q.json"
# Fallback valid URL if the above fails (using a known simple one for test stability if needed)
# But we'll try a fresh one.
# Alternative: "https://assets9.lottiefiles.com/packages/lf20_aa0wy04q.json" (AI Robot)

lottie_robot_json = load_lottieurl(LOTTIE_ROBOT)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Inter:wght@300;400;600&display=swap');

        /* GLOBAL THEME */
        :root {
            --bg-dark: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.6);
            --primary: #06b6d4;  /* Cyan */
            --accent: #ec4899;   /* Pink/Magenta */
            --rotten: #ef4444;
            --fresh: #22c55e;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--text-main);
        }
        
        .stApp {
            background: linear-gradient(135deg, #020617 0%, #172554 100%);
            background-attachment: fixed;
        }

        h1, h2, h3, .brand-font {
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.05em;
        }

        /* CARD STYLES */
        .glass-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 20px;
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(6, 182, 212, 0.15);
            border-color: rgba(6, 182, 212, 0.3);
        }

        /* ANIMATIONS */
        @keyframes scanline {
            0% { transform: translateY(-100%); }
            100% { transform: translateY(100%); }
        }
        
        .scanner-line {
            position: absolute;
            top: 0; left: 0; right: 0; height: 4px;
            background: linear-gradient(to right, transparent, var(--primary), transparent);
            opacity: 0.5;
            animation: scanline 2.5s linear infinite;
            pointer-events: none;
        }

        @keyframes pulse-glow {
            0% { box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.4); }
            70% { box-shadow: 0 0 0 15px rgba(6, 182, 212, 0); }
            100% { box-shadow: 0 0 0 0 rgba(6, 182, 212, 0); }
        }

        .status-dot {
            height: 12px; width: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-active {
            background-color: var(--primary);
            animation: pulse-glow 2s infinite;
        }
        .status-rotten { background-color: var(--rotten); box-shadow: 0 0 10px var(--rotten); }
        .status-fresh { background-color: var(--fresh); box-shadow: 0 0 10px var(--fresh); }

        /* BUTTONS */
        div[data-testid="stButton"] button {
            background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            font-family: 'Orbitron', sans-serif;
            transition: all 0.3s ease;
        }
        div[data-testid="stButton"] button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.5);
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background-color: rgba(2, 6, 23, 0.95);
            border-right: 1px solid rgba(255,255,255,0.05);
        }

        /* SCROLLBAR */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }

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
            st.error(f"YOLO model file not found: {path}", icon="🚨")
            st.stop()
        return YOLO(path)
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}", icon="💥")
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
    det_id = str(uuid.uuid4())[:8]
    try:
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
        x1, y1, x2, y2 = xyxy
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
# 3. SIDEBAR (System Control)
# -----------------------------
with st.sidebar:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if lottie_robot_json:
        st_lottie(lottie_robot_json, height=150, key="sidebar_anim")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='color: #06b6d4; font-size: 24px; margin-bottom: 0;'>FRUIT GUARD</h1>
            <div style='font-size: 10px; letter-spacing: 2px; color: #94a3b8; text-transform: uppercase;'>AI Vision System v2.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ System Config")
    model_path = st.text_input("Model Path", "best.pt")
    conf_threshold = st.slider("Sensitivity (Confidence)", 0.1, 1.0, 0.45)

    st.markdown("### 💾 Data Log")
    save_only_above = st.slider("Auto-Save Threshold", 0.1, 1.0, 0.80)
    cooldown_sec = st.slider("Capture Cooldown (s)", 0.0, 10.0, 2.0, 0.5)

    st.markdown("---")
    if st.button("FORMAT STORAGE", use_container_width=True):
        n = clear_storage()
        st.toast(f"System Purged: {n} files deleted.", icon="🗑️")
        time.sleep(1)
        st.rerun()

    st.markdown(
        """
        <div style='margin-top: 20px; padding: 10px; border-radius: 8px; background: rgba(255,255,255,0.05); font-size: 0.8em; color: #64748b;'>
        <strong>Classes:</strong><br>
        • Fresh: apple, banana, orange<br>
        • Rotten: apple, banana, orange
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# 4. MAIN WORKSPACE
# -----------------------------

# Header Area with stats or welcome
col_head1, col_head2 = st.columns([2, 1])
with col_head1:
    st.markdown(
        "<h1 style='font-size: 3em;'>LIVE OPS CENTER</h1>", unsafe_allow_html=True
    )
with col_head2:
    pass


col_vid, col_ai = st.columns([1.5, 1], gap="large")

with col_vid:
    st.markdown("### 📡 Visual Feed Input")

    # Custom Tab-like toggle
    input_mode = st.radio(
        "Source",
        ["Upload File", "Sample Library"],
        horizontal=True,
        label_visibility="collapsed",
    )

    video_file = None
    selected_sample_path = None

    if input_mode == "Upload File":
        video_file = st.file_uploader(
            "", type=["mp4", "mov", "avi"], label_visibility="collapsed"
        )
    else:
        sample_dir = Path("sample_videos")
        if sample_dir.exists():
            sample_files = sorted([f.name for f in sample_dir.glob("*.mp4")])
            selected_sample = st.selectbox("Select Sample Stream", sample_files)
            if selected_sample:
                selected_sample_path = sample_dir / selected_sample
        else:
            st.warning("Library 'sample_videos' offline.")

    video_screen = st.empty()

    c_ctrl1, c_ctrl2 = st.columns(2)

    is_ready = (input_mode == "Upload File" and video_file is not None) or (
        input_mode == "Sample Library" and selected_sample_path is not None
    )

    start_btn = c_ctrl1.button(
        "▶ INITIATE SCAN",
        use_container_width=True,
        disabled=not is_ready,
        type="primary",
    )
    stop_btn = c_ctrl2.button("⏹ TERMINATE", use_container_width=True)

    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False

    if stop_btn:
        st.session_state.stop_requested = True
        st.rerun()

    if start_btn and is_ready:
        st.session_state.stop_requested = False
        st.session_state.processed_ids = set()

        if not Path(model_path).exists():
            st.error(f"Neural Network Weights Missing: {model_path}")
            st.stop()

        final_video_path = None
        if input_mode == "Upload File":
            tmp_video_path = (
                TMP_DIR / f"upload_{uuid.uuid4().hex[:8]}_{Path(video_file.name).name}"
            )
            with open(tmp_video_path, "wb") as f:
                f.write(video_file.read())
            final_video_path = tmp_video_path
        else:
            final_video_path = selected_sample_path

        cap = cv2.VideoCapture(str(final_video_path))
        model = load_yolo_model(model_path)

        while cap.isOpened():
            if st.session_state.stop_requested:
                st.warning("Process Aborted by User.")
                break

            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame, conf=conf_threshold, persist=True, verbose=False
            )

            # Processing Logic
            for r in results:
                for box in r.boxes:
                    if box.id is not None:
                        track_id = int(box.id.item())
                        if track_id not in st.session_state.processed_ids:
                            cls_id = int(box.cls[0].item())
                            label = model.names.get(cls_id, str(cls_id))
                            conf = _safe_float(box.conf[0])

                            if conf >= save_only_above:
                                if save_detection(frame, box, label, conf):
                                    st.session_state.processed_ids.add(track_id)

            # Display
            annotated_frame = results[0].plot()
            video_screen.empty()
            with video_screen.container():
                st.markdown(
                    """
                    <div style="position: relative; border: 2px solid #06b6d4; border-radius: 12px; overflow: hidden; box-shadow: 0 0 20px rgba(6, 182, 212, 0.2);">
                        <div class="scanner-line"></div>
                        <div style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); padding: 4px 8px; border-radius: 4px; color: #06b6d4; font-family: 'Orbitron'; font-size: 0.8em; z-index: 10;">
                            LIVE FEED • REC
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Sequence Complete.")


# -----------------------------
# 5. ANALYSIS PANEL
# -----------------------------
with col_ai:
    st.markdown("### 🧬 Analysis Hub")

    active = st.session_state.active_det

    if active:
        # Highlight logic
        is_rotten = "rotten" in active.get("label", "").lower()
        border_col = "#ef4444" if is_rotten else "#22c55e"
        glow_col = "rgba(239, 68, 68, 0.4)" if is_rotten else "rgba(34, 197, 94, 0.4)"
        status_text = "CRITICAL" if is_rotten else "OPTIMAL"

        st.markdown(
            f"""
            <div class="glass-card" style="border: 1px solid {border_col}; box-shadow: 0 0 30px {glow_col};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <div style="font-family: 'Orbitron'; font-size: 1.2em; font-weight: bold;">OBJECT #{active.get("id")}</div>
                    <div style="background: {border_col}; color: #000; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em;">{status_text}</div>
                </div>
                <div style="display: flex; gap: 15px;">
                    <div style="flex: 1;">
                        <img src="app/active_img" style="width: 100%; border-radius: 8px; display: none;" /> 
                        <!-- Image handled by st.image below for security/path ease -->
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c_img, c_det = st.columns([1, 1.2])
        apath = active.get("image_path", "")

        with c_img:
            if apath and Path(apath).exists():
                st.image(apath, use_container_width=True)

        with c_det:
            st.markdown(f"**Class:** `{active.get('label').upper()}`")
            st.markdown(f"**Confidence:** `{float(active.get('confidence', 0)):.1%}`")
            st.markdown(f"**Timestamp:** `{active.get('timestamp')}`")

        # Chat Interface
        with st.expander("💬 AI FORENSICS", expanded=True):
            gemini = get_gemini_model()
            if not gemini:
                st.error("AI Module Offline (Missing API Key)")
            else:
                chat_id = f"chat_{active.get('id')}"
                if chat_id not in st.session_state:
                    st.session_state[chat_id] = []

                # Chat history
                for m in st.session_state[chat_id]:
                    with st.chat_message(m["role"]):
                        st.write(m["content"])

                if prompt := st.chat_input("Query analysis module..."):
                    st.session_state[chat_id].append(
                        {"role": "user", "content": prompt}
                    )
                    st.rerun()

                # Response generation
                if (
                    st.session_state[chat_id]
                    and st.session_state[chat_id][-1]["role"] == "user"
                ):
                    with st.spinner("Processing neural request..."):
                        try:
                            last_p = st.session_state[chat_id][-1]["content"]
                            if apath and Path(apath).exists():
                                resp = gemini.generate_content(
                                    [
                                        f"System Alert: Object detected as {active.get('label')}. User Query: {last_p}",
                                        Image.open(apath).convert("RGB"),
                                    ]
                                )
                                answer = resp.text
                            else:
                                answer = "Visual data corrupted/missing."
                        except Exception as e:
                            answer = f"System Error: {e}"

                        st.session_state[chat_id].append(
                            {"role": "assistant", "content": answer}
                        )
                        st.rerun()

    else:
        st.info("Awaiting Selection... Initiate scan or select from history.", icon="ℹ️")

    st.markdown("### 🖼️ Detection Gallery")

    # History Grid
    all_metas = load_detection_metas(limit=10)
    if not all_metas:
        st.write("Gallery Empty")
    else:
        with st.container(height=500):
            # Responsive Grid Layout
            grid_cols = st.columns(2)
            for i, meta in enumerate(all_metas):
                col_idx = i % 2
                with grid_cols[col_idx]:
                    is_rot = "rotten" in meta.get("label", "").lower()

                    with st.container(border=True):
                        # Custom mini-card
                        ip = meta.get("image_path")
                        if ip and Path(ip).exists():
                            st.image(ip, use_container_width=True)

                        st.markdown(f"**{meta.get('label', 'Unknown').upper()}**")
                        st.caption(f"ID: {meta.get('id')} | {meta.get('timestamp')}")

                        if st.button(
                            "INSPECT", key=f"btn_{meta['id']}", use_container_width=True
                        ):
                            st.session_state.active_det = meta
                            st.rerun()
