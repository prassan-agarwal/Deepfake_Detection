import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_loader import DeepfakeDataset
from models.hybrid_model import DeepfakeHybridModel

import albumentations as A
from albumentations.pytorch import ToTensorV2

class VideoAlbumentationsTransform:
    def __init__(self, is_train=True):
        self.is_sequence_transform = True
        if is_train:
            self.transform = A.ReplayCompose([
                A.ImageCompression(p=0.2), # Mild compression
                A.GaussNoise(p=0.1), # Very subtle sensor noise
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2),
                A.HorizontalFlip(p=0.5),
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.ReplayCompose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
    def __call__(self, frames):
        # frames is a list of HWC numpy arrays
        augmented_frames = []
        if not frames:
            return augmented_frames
            
        # Apply to first frame and record the deterministic replay parameters
        data = self.transform(image=frames[0])
        augmented_frames.append(data['image'])
        
        # Replay EXACT same parameters to all subsequent frames in video
        for frame in frames[1:]:
            res = A.ReplayCompose.replay(data['replay'], image=frame)
            augmented_frames.append(res['image'])
            
        return augmented_frames

def get_transforms(is_train=True):
    return VideoAlbumentationsTransform(is_train=is_train)

def train_model(real_dir, fake_dir, num_epochs=20, batch_size=2, accumulation_steps=4, sequence_length=16):
    
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Optional but highly recommended for RTX 3050 Ti:
    # torch.backends.cudnn.benchmark = True
    
    # 2. Dataset and DataLoader setup
    print("Loading dataset for dynamic Train/Val 70/30 split...")
    import copy
    from torch.utils.data import Subset
    
    # Load entire dataset structure first
    master_dataset = DeepfakeDataset(real_dir=real_dir, fake_dir=fake_dir, sequence_length=sequence_length, transform=None)
    
    # Shuffle and split 70% Train, 30% Val
    dataset_size = len(master_dataset)
    indices = torch.randperm(dataset_size).tolist()
    train_size = int(0.7 * dataset_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Clone datasets so they can safely have independent transforms (Heavy augmentations for Train, Clean for Val)
    train_dataset = copy.deepcopy(master_dataset)
    train_dataset.transform = get_transforms(is_train=True)
    
    val_dataset = copy.deepcopy(master_dataset)
    val_dataset.transform = get_transforms(is_train=False)
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Extract labels to compute sample weights for the WeightedRandomSampler inside the Training Split ONLY
    labels = [train_dataset.video_frames[i][1] for i in train_indices]
    num_real = labels.count(0)
    num_fake = labels.count(1)
    
    class_weights = {0: 1.0 / max(1, num_real), 1: 1.0 / max(1, num_fake)}
    sample_weights = [class_weights[label] for label in labels]
    
    # WeightedRandomSampler ensures mathematically even batches of Real/Fakes during training
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_subset), replacement=True)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Split completed | Train: {len(train_subset)} (Real: {num_real}, Fake: {num_fake}) | Val: {len(val_subset)}")
    
    # 3. Model Initialization
    # Make sure to import this from the appropriate file
    model = DeepfakeHybridModel()
    model = model.to(device)
    
    # 4. Loss and Optimizer
    print(f"Dataset split: Real={num_real}, Fake={num_fake}")
    
    # Because we implemented the WeightedRandomSampler, batches are perfectly balanced. 
    # The blunt pos_weight is completely removed so the loss formula is naturally unbiased!
    criterion = nn.BCEWithLogitsLoss()
    
    # Differential Learning Rates (Catastrophic Forgetting Fix)
    # The DeiT backbone is pretrained; we fine-tune it gently. The randomly initialized heads learn aggressively.
    optimizer = torch.optim.AdamW([
        {'params': model.spatial.parameters(), 'lr': 1e-5},
        {'params': model.temporal.parameters(), 'lr': 1e-3},
        {'params': model.frequency.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)
    
    # Dynamic Learning Rate Scheduler (slices LR by 50% if Accuracy plateaus for 2 epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # 5. Mixed Precision Setup
    # This scaler handles FP16 gradients to prevent underflow
    scaler = torch.cuda.amp.GradScaler()
    
    # 6. Training Loop
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # We start with cleared gradients
        optimizer.zero_grad()
        
        # Use tqdm for progress bar
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (sequences, labels) in progress_bar:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1) # [B, 1]
            
            # Forward pass with Automatic Mixed Precision
            with torch.cuda.amp.autocast():
                logits = model(sequences)
                loss = criterion(logits, labels)
                
                # Normalize loss to account for gradient accumulation
                loss = loss / accumulation_steps
                
            # Backward pass (scaled)
            scaler.scale(loss).backward()
            
            # Update weights only after accumulation_steps have passed
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Step optimizer and update scaler
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True) # Slightly more memory efficient
            
            # Tracking metrics
            running_loss += loss.item() * accumulation_steps # Un-normalize for display
            
            # Calculate accuracy
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # Formatting progress bar
            progress_bar.set_postfix({
                'Loss': f"{running_loss / (batch_idx + 1):.4f}", 
                'Acc': f"{correct_predictions / total_samples:.4f}"
            })
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_samples
        print(f"Epoch {epoch+1} Train | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
        
        # ==========================================
        # --- UNSEEN VALIDATION LOOP ---
        # ==========================================
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): # Disable gradients to stop data leakage
            for sequences, labels in tqdm(val_loader, desc=f"Val {epoch+1}/{num_epochs}", leave=False):
                sequences = sequences.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).unsqueeze(1)
                
                with torch.cuda.amp.autocast():
                    logits = model(sequences)
                    loss = criterion(logits, labels)
                    
                val_running_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} VALIDATION | Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}\n")
        
        # Step the Learning Rate Scheduler purely based on Unseen VALIDATION Accuracy
        scheduler.step(val_acc)
        
        # Early-Stopping Logic purely on VALIDATION Accuracy instead of Train metrics
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_hybrid_model.pth")
            print(f">>> Saved new absolute best model checkpoint! (Val Acc: {val_acc:.4f})")

if __name__ == "__main__":
    
    # Update these paths to point to your processed local dataset
    REAL_DIR = "dataset/processed/real"
    FAKE_DIR = "dataset/processed/fake"
    
    # Run the training loop!
    train_model(
        real_dir=REAL_DIR, 
        fake_dir=FAKE_DIR, 
        num_epochs=15,      # Increased to 15 epochs for new heavily-augmented learning pipeline
        batch_size=2,       # Keep very small (2) for 4GB VRAM
        accumulation_steps=4, # Simulates batch_size of 8
        sequence_length=32  # Load 32 evenly spaced frames per video
    )
