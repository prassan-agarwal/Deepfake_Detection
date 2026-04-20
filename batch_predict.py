import os
import cv2
import torch
import numpy as np
from mtcnn import MTCNN
from torchvision import transforms
from tqdm import tqdm
import csv
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_model import DeepfakeHybridModel


def extract_faces(video_path, detector, transform, sequence_length=16):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)

    frames = []

    for idx in indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        if len(faces) > 0:

            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            face = rgb[y:y+h, x:x+w]

        else:

            h, w, _ = rgb.shape
            m = min(h, w)
            y0 = (h-m)//2
            x0 = (w-m)//2
            face = rgb[y0:y0+m, x0:x0+m]

        face = transform(face)
        frames.append(face)

    cap.release()

    if len(frames) == 0:
        return None

    while len(frames) < sequence_length:
        frames.append(frames[-1])

    return torch.stack(frames)


def batch_predict(folder, model_path="best_hybrid_model.pth", sequence_length=16):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Loading model...")

    model = DeepfakeHybridModel()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()

    detector = MTCNN()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    videos = [v for v in os.listdir(folder) if v.endswith(".mp4")]
    videos.sort()

    real_count = 0
    fake_count = 0

    results = []

    print("\nStarting batch detection...\n")

    for video in tqdm(videos):

        video_path = os.path.join(folder, video)

        frames = extract_faces(video_path, detector, transform, sequence_length)

        if frames is None:
            continue

        input_tensor = frames.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits).item()

        is_fake = prob >= 0.5
        confidence = prob if is_fake else (1-prob)

        label = "FAKE" if is_fake else "REAL"

        if is_fake:
            fake_count += 1
            emoji = "🚨"
        else:
            real_count += 1
            emoji = "✅"

        print(f"{video} → {emoji} {label} ({confidence*100:.2f}%)")

        results.append([video, label, confidence*100])

    print("\n========= SUMMARY =========")
    print("Total videos :", len(results))
    print("Real videos  :", real_count)
    print("Fake videos  :", fake_count)
    print("===========================\n")

    with open("batch_results.csv","w",newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["Video","Prediction","Confidence"])
        writer.writerows(results)

    print("Saved results → batch_results.csv")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("python inference/batch_detect_fast.py <video_folder>")
    else:
        batch_predict(sys.argv[1])