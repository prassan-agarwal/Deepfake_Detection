# 🔍 Deepfake Detector

A deep learning-based video deepfake detection system using a **Hybrid Multi-Branch Architecture** (Spatial + Temporal + Frequency analysis) with a FastAPI backend and Next.js frontend.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14+-000000?logo=next.js)

---

## 🏗️ Architecture

The model uses a **three-branch hybrid architecture** that fuses features from multiple analysis domains:

```
Video Input (MP4/AVI) → Frame Sampling (32 frames) → MTCNN Face Crop (224×224)
                                                                │
              ┌─────────────────────────────────────────────────┤
              │                         │                       │
              ▼                         ▼                       ▼
     Spatial Branch            Temporal Branch        Frequency Branch
   (DeiT-Small ViT)           (1-layer BiLSTM)        (DFT Matrix + CNN)
    12 Transformer Blocks       Bidirectional          ONNX-compatible
    CLS token → 384-d → 256-d   256-d → 256-d         2D DFT → 128-d
              │                         │                       │
              └─────────────────────────┴───────────────────────┘
                                        ▼
                      Late Fusion: concat [256 + 256 + 128] = 640-d
                                        │
                               FC(640→256) → FC(256→1)
                                        │
                               BCEWithLogitsLoss
                                  (Real / Fake)
```

| Branch | Backbone | Output Dim | Purpose |
|--------|----------|------------|---------|
| **Spatial** | DeiT-Small (`deit_small_patch16_224`) | 256-d | Per-frame ViT features via CLS token |
| **Temporal** | 1-layer Bidirectional LSTM | 256-d (128×2) | Inter-frame consistency across 32-frame sequences |
| **Frequency** | DFT matrix multiply + 3-layer CNN | 128-d | Spectral anomalies — ONNX-safe (no `torch.fft`) |

---

## 📁 Project Structure

```
Deepfake_Detector-Anti/
├── models/                 # Model architecture definitions
├── preprocessing/          # Data preprocessing pipeline
│   ├── extract_frames.py   # Video → frame extraction
│   ├── face_detection.py   # MTCNN face detection & cropping
│   └── process_dataset.py  # Full dataset processing
├── training/
│   └── train.py                # Training loop (AMP, gradient accumulation, checkpoint)
├── inference/                  # Inference & evaluation
│   ├── detect_single_video.py  # CLI: single video detection
│   ├── evaluate.py             # Metrics (AUC, accuracy, confusion matrix)
│   ├── export_onnx.py          # ONNX model export
│   └── gradcam.py              # GradCAM attention visualizations
├── utils/
│   └── dataset_loader.py       # PyTorch Dataset & DataLoader
├── backend/                    # FastAPI REST API
│   ├── main.py                 # API server & endpoints
│   └── inference.py            # Backend inference logic
├── frontend/                   # Next.js + TypeScript web interface
│   └── src/
├── app/
│   └── app.py                  # Streamlit demo interface
├── batch_predict.py            # Batch video prediction script
├── analyze_results.py          # Results analysis & plotting
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-compatible GPU (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Deepfake_Detector-Anti.git
cd Deepfake_Detector-Anti
```

### 2. Set Up Python Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 3. Train the Model

Place your dataset videos in `dataset/raw/real/` and `dataset/raw/fake/`, then run:

```bash
python -m training.train
```

Key training details:
- **Sequence length**: 32 evenly-sampled frames per video
- **Batch size**: 2 with 4-step gradient accumulation (effective batch size = 8)
- **Mixed Precision**: AMP (`torch.cuda.amp`) for RTX GPU memory efficiency
- **Optimizer**: AdamW with differential LRs — `1e-5` for DeiT backbone, `1e-3` for heads
- **Scheduler**: `ReduceLROnPlateau` on validation accuracy
- **Split**: 70% train / 30% validation with `WeightedRandomSampler` for class balance
- **Checkpoint**: Best model saved to `best_hybrid_model.pth` based on validation accuracy

### 4. Run the Backend (FastAPI)

```bash
python -m backend.main
```

The API will be available at `http://localhost:8000`.

### 5. Run the Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## 📊 Evaluation Results

The model generates the following evaluation artifacts in `inference_results/`:

- **Confusion Matrix** — classification performance breakdown
- **ROC Curve** — receiver operating characteristic analysis
- **GradCAM Visualizations** — model attention heatmaps on real vs fake frames

Trained on YouTube-Real + Celeb-DF v2:

| Metric | Score |
|--------|-------|
| **Accuracy** | ~85–90% |
| **AUC-ROC** | ~0.90+ |
| **Sequence Length** | 32 frames |
| **Training Split** | 70% train / 30% val |
| **Datasets** | YouTube-Real + Celeb-DF v2 |

---

## 🔌 API Reference

### `POST /api/detect`

Upload a video for deepfake detection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `video` | `UploadFile` | Video file (`.mp4`, `.avi`, `.mov`) |

**Response:**
```json
{
  "success": true,
  "filename": "video.mp4",
  "is_fake": true,
  "fake_probability": 0.92,
  "confidence_percentage": "92.0%"
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Spatial Branch** | DeiT-Small ViT (`deit_small_patch16_224`) via `timm` |
| **Temporal Branch** | 1-layer Bidirectional LSTM |
| **Frequency Branch** | DFT matrix multiply + 3-layer CNN (ONNX-safe) |
| **Face Detection** | MTCNN via `facenet-pytorch` (GPU-accelerated) |
| **Training** | PyTorch, AMP, AdamW, WeightedRandomSampler |
| **Backend API** | FastAPI, Uvicorn |
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS |
| **Demo App** | Streamlit |
| **Visualization** | GradCAM, Matplotlib, Seaborn |
| **Model Export** | ONNX (DFT matrix-multiply for compatibility) |

---

## 📄 License

This project was developed as a **B.Tech Capstone Project** and is intended for educational and research purposes.

---

## 🙏 Acknowledgements

- [Celeb-DF v2 Dataset](https://github.com/yuezunli/celeb-deepfakeforensics)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) for GPU-accelerated MTCNN
- [timm](https://github.com/huggingface/pytorch-image-models) for DeiT-Small backbone
