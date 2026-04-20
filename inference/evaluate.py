import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_model import DeepfakeHybridModel
from utils.dataset_loader import DeepfakeDataset

from training.train import get_transforms

def evaluate_model(real_dir, fake_dir, model_path="best_hybrid_model.pth", sequence_length=16, batch_size=2):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    # 1. Load Model
    model = DeepfakeHybridModel()
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval() # CRITICAL: Set model to evaluation mode
    
    # 2. Dataset and DataLoader setup
    transform = get_transforms(is_train=False)
    
    print("Loading evaluation dataset...")
    # NOTE: In a real-world scenario, you should use a separate Validation or Test split here.
    # Currently reusing the processed folder for demonstration.
    dataset = DeepfakeDataset(real_dir=real_dir, fake_dir=fake_dir, sequence_length=sequence_length, transform=transform)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Total sequences to evaluate: {len(dataset)}")
    
    # 3. Evaluation Loop
    all_preds = []
    all_probs = []
    all_labels = []
    
    progress_bar = tqdm(eval_loader, desc="Evaluating")
    
    with torch.no_grad(): # No gradients needed for evaluation (saves memory)
        for sequences, labels in progress_bar:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            # Note: During evaluation, we don't necessarily need mixed precision, but it's fine
            logits = model(sequences) 
            probs = torch.sigmoid(logits).squeeze()
            
            # Handle single-element batches
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
                
            preds = (probs > 0.5).int()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Calculate Metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    
    print("\n" + "="*40)
    print("Deepfake Detection Model Evaluation")
    print("="*40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}  (Rate of true fake detections out of all predicted fakes)")
    print(f"Recall:    {recall:.4f}  (Rate of fakes successfully detected out of all actual fakes)")
    print(f"F1-Score:  {f1:.4f}  (Harmonic mean of precision and recall)")
    print(f"AUC-ROC:   {auc_roc:.4f}  (Model's ability to distinguish classes overall)")
    print("="*40)
    
    # 5. Generate Visualizations
    os.makedirs("inference_results", exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('inference_results/confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('inference_results/roc_curve.png')
    plt.close()
    
    print("\nVisualizations saved:")
    print(" - inference_results/confusion_matrix.png")
    print(" - inference_results/roc_curve.png")

if __name__ == "__main__":
    REAL_DIR = "dataset/processed/real"
    FAKE_DIR = "dataset/processed/fake"
    
    evaluate_model(real_dir=REAL_DIR, fake_dir=FAKE_DIR, sequence_length=32)
