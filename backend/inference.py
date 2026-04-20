import os
import cv2
import numpy as np
from mtcnn import MTCNN
import onnxruntime as ort

# Project root directory (one level up from backend/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize MTCNN and ONNX Session once globally to avoid overhead on every request
detector = MTCNN()

# Load the ONNX Model
MODEL_PATH = os.path.join(PROJECT_ROOT, "inference_results", "deepfake_model.onnx")
if os.path.exists(MODEL_PATH):
    ort_session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
else:
    print(f"WARNING: Model {MODEL_PATH} not found. Please train and export the model first.")
    ort_session = None

def get_normalize_transform():
    # Same normalizations used during training
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return mean, std

def extract_and_process_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
        
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    processed_faces = []
    
    mean, std = get_normalize_transform()
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # If standard extraction fails, just grab the next sequential frame
            ret, frame = cap.read()
            if not ret:
                continue
                
        # Detect Face using MTCNN
        # Convert BGR (OpenCV) to RGB (MTCNN expects RGB, model expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)
        
        if len(faces) == 0:
            # Fallback: Just take the center crop of the whole frame if no face found
            h, w, _ = rgb_frame.shape
            size = min(h, w)
            y, x = (h - size) // 2, (w - size) // 2
            face = rgb_frame[y:y+size, x:x+size]
        else:
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            face = rgb_frame[y:y+h, x:x+w]
            
        # Resize to exactly 224x224
        face = cv2.resize(face, (224, 224))
        
        # Normalize (To PIL transform -> ToTensor equivalents)
        # ToTensor scales [0, 255] down to [0.0, 1.0]
        face_normalized = face.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization (channel-wise)
        face_normalized = (face_normalized - mean) / std
        
        # Re-arrange to Channel First: [C, H, W] instead of [H, W, C]
        face_normalized = np.transpose(face_normalized, (2, 0, 1))
        
        processed_faces.append(face_normalized)
        
    cap.release()
    
    # We need exactly `num_frames` (e.g., 16). 
    # If the video is extremely short or corrupt and gave us fewer frames, pad it out.
    while len(processed_faces) < num_frames and len(processed_faces) > 0:
        processed_faces.append(processed_faces[-1]) # Duplicate last frame
        
    if len(processed_faces) == 0:
        return None
        
    # Stack into [Sequence, Channels, Height, Width]
    sequence_array = np.stack(processed_faces, axis=0)
    
    # Add batch dimension: [1, Sequence, Channels, Height, Width]
    sequence_array = np.expand_dims(sequence_array, axis=0)
    
    return sequence_array

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

import torch
import base64
from inference.gradcam import GradCAM, overlay_cam
from models.hybrid_model import DeepfakeHybridModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pytorch_model = None

def load_pytorch_model():
    global pytorch_model
    if pytorch_model is None:
        model_path = os.path.join(PROJECT_ROOT, "best_hybrid_model.pth")
        if os.path.exists(model_path):
            pytorch_model = DeepfakeHybridModel()
            pytorch_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            pytorch_model = pytorch_model.to(device)
            pytorch_model.eval()
        else:
            print("WARNING: PyTorch model not found for GradCAM generation.")
    return pytorch_model

def get_normalize_transform():
    # Same normalizations used during training
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return mean, std

def run_deepfake_inference(video_path):
    if ort_session is None:
        raise RuntimeError("ONNX Model strictly required but not loaded.")
        
    # 1. Prepare Data
    input_tensor = extract_and_process_frames(video_path, num_frames=32)
    
    if input_tensor is None:
        raise ValueError("Could not extract any valid sequence from the video.")
        
    # 2. Run ONNX Inference
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    # Run the model
    # Note: ONNX requires numpy arrays of float32
    outputs = ort_session.run([output_name], {input_name: input_tensor})
    
    # 3. Postprocess
    logit = outputs[0][0][0] # Access single output scalar logit
    probability = float(sigmoid(logit))
    
    return probability, input_tensor

def generate_gradcam_base64(sequence_array):
    model = load_pytorch_model()
    if model is None:
        return None
        
    target_layer = model.spatial.backbone.blocks[-1]
    cam_extractor = GradCAM(model, target_layer, is_transformer=True, grid_size=14)
    
    # sequence_array is [1, 16, 3, 224, 224] as numpy float32.
    input_tensor = torch.tensor(sequence_array).to(device)
    
    cam, _ = cam_extractor.generate(input_tensor)
    
    # Select middle frame to overlay
    mid_idx = sequence_array.shape[1] // 2
    
    frame = sequence_array[0, mid_idx]
    mean, std = get_normalize_transform()
    frame = frame.transpose(1, 2, 0) # [224, 224, 3]
    frame = (frame * std) + mean
    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    
    heatmap = cam[mid_idx]
    
    overlaid = overlay_cam(frame, heatmap, alpha=0.5)
    
    # Convert RGB to BGR for cv2 encoding
    overlaid_bgr = cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR)
    
    # Encode to JPEG, then Base64
    success, buffer = cv2.imencode('.jpg', overlaid_bgr)
    if not success:
        return None
        
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"
