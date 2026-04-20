import os
import cv2
import numpy as np
import streamlit as st
import tempfile
import onnxruntime as ort
from mtcnn import MTCNN

# ─── Config ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "inference_results", "deepfake_model.onnx")
NUM_FRAMES = 16
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── Model Loading (cached so it loads only once) ────────────────────────────
@st.cache_resource
def load_model():
    """Load ONNX model and MTCNN detector once and cache them."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at `{MODEL_PATH}`. Please export the ONNX model first.")
        st.stop()
    
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    detector = MTCNN()
    return session, detector


# ─── Inference Helpers ────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def extract_and_process_frames(video_path: str, detector: MTCNN) -> np.ndarray | None:
    """Extract faces from sampled frames, normalize, and return as a batch tensor."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    processed_faces = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            ret, frame = cap.read()
            if not ret:
                continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)

        if faces:
            x, y, w, h = faces[0]["box"]
            x, y = max(0, x), max(0, y)
            face = rgb_frame[y : y + h, x : x + w]
        else:
            # Fallback: center crop
            h, w, _ = rgb_frame.shape
            size = min(h, w)
            sy, sx = (h - size) // 2, (w - size) // 2
            face = rgb_frame[sy : sy + size, sx : sx + size]

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype(np.float32) / 255.0
        face = (face - IMAGENET_MEAN) / IMAGENET_STD
        face = np.transpose(face, (2, 0, 1))  # HWC → CHW
        processed_faces.append(face)

    cap.release()

    if not processed_faces:
        return None

    # Pad to NUM_FRAMES if needed
    while len(processed_faces) < NUM_FRAMES:
        processed_faces.append(processed_faces[-1])

    # Shape: [1, NUM_FRAMES, 3, 224, 224]
    return np.expand_dims(np.stack(processed_faces, axis=0), axis=0)


def run_inference(video_path: str, session: ort.InferenceSession, detector: MTCNN) -> dict:
    """Run full inference pipeline and return structured results."""
    input_tensor = extract_and_process_frames(video_path, detector)

    if input_tensor is None:
        return {"error": "Could not extract frames from video. It may be corrupted or too short."}

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_tensor})

    logit = outputs[0][0][0]
    probability = float(sigmoid(logit))
    is_fake = probability >= 0.5
    confidence = probability if is_fake else (1.0 - probability)

    return {
        "is_fake": is_fake,
        "probability": probability,
        "confidence": confidence,
    }


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Deepfake Video Detector",
        page_icon="🕵️",
        layout="centered",
    )

    # ── Custom CSS ──
    st.markdown("""
    <style>
        .main-title {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 0;
        }
        .subtitle {
            text-align: center;
            color: #888;
            font-size: 1.1rem;
            margin-top: -10px;
            margin-bottom: 30px;
        }
        .result-card {
            padding: 24px;
            border-radius: 16px;
            text-align: center;
            margin: 20px 0;
        }
        .result-fake {
            background: linear-gradient(135deg, #ff416c22, #ff4b2b22);
            border: 2px solid #ff416c;
        }
        .result-real {
            background: linear-gradient(135deg, #00b09b22, #96c93d22);
            border: 2px solid #00b09b;
        }
        .confidence-text {
            font-size: 2.5rem;
            font-weight: 700;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">🕵️ Deepfake Video Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a video to analyze if it\'s real or AI-generated</p>', unsafe_allow_html=True)

    # Load model
    session, detector = load_model()

    # ── File Upload ──
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"],
        help="Supported formats: MP4, AVI, MOV (max 200 MB)",
    )

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            with st.spinner("Extracting faces and running inference... This may take a moment."):
                # Save to temp file (OpenCV needs a file path)
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                temp_path = tfile.name
                tfile.close()

                try:
                    result = run_inference(temp_path, session, detector)
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

            # ── Display Results ──
            if "error" in result:
                st.error(f"⚠️ {result['error']}")
            else:
                is_fake = result["is_fake"]
                confidence = result["confidence"]
                probability = result["probability"]

                if is_fake:
                    st.markdown(f"""
                    <div class="result-card result-fake">
                        <h2>🚨 DEEPFAKE DETECTED</h2>
                        <p class="confidence-text" style="color: #ff416c;">{confidence * 100:.1f}% Confidence</p>
                        <p>Fake Probability: {probability:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card result-real">
                        <h2>✅ VIDEO APPEARS REAL</h2>
                        <p class="confidence-text" style="color: #00b09b;">{confidence * 100:.1f}% Confidence</p>
                        <p>Fake Probability: {probability:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Detail columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Verdict", "FAKE 🚨" if is_fake else "REAL ✅")
                with col2:
                    st.metric("Confidence", f"{confidence * 100:.1f}%")
                with col3:
                    st.metric("Fake Prob", f"{probability:.4f}")

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#666; font-size:0.85rem;'>"
        "Powered by Hybrid Deep Learning (MobileNetV3 + BiLSTM + FFT) • ONNX Runtime"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
