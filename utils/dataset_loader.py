import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict


class DeepfakeDataset(Dataset):

    def __init__(self, real_dir, fake_dir, sequence_length=16, transform=None):
        self.sequence_length = sequence_length
        self.transform = transform
        
        self.video_frames = [] # List of tuples: (list_of_frame_paths, label)
        
        self._load_video_frames(real_dir, label=0)
        self._load_video_frames(fake_dir, label=1)

    def _load_video_frames(self, directory, label):
        # Group frames by their source video
        # The naming convention is typically videoName_frameNumber.jpg
        video_groups = defaultdict(list)
        for img_name in sorted(os.listdir(directory)):
            if not img_name.endswith('.jpg'):
                continue
                
            # Extract video base name (e.g., '00000' from '00000_15.jpg')
            # Example fake name: 'id0_id16_0000_15.jpg' -> base depends on format
            # A safe way is to split by '_' and drop the last part (frame number)
            parts = img_name.rsplit('_', 1)
            if len(parts) == 2:
                video_name = parts[0]
                img_path = os.path.join(directory, img_name)
                video_groups[video_name].append(img_path)
                
        # Filter and store sequences
        for video_name, frame_paths in video_groups.items():
            # Sort by frame number extracted from the end of the filename
            frame_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            
            if len(frame_paths) >= self.sequence_length:
                # We can take evenly spaced frames or just the first N
                # Let's take evenly spaced to cover the whole video temporally
                indices = np.linspace(0, len(frame_paths) - 1, self.sequence_length, dtype=int)
                selected_frames = [frame_paths[i] for i in indices]
                self.video_frames.append((selected_frames, label))

    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, idx):
        frame_paths, label = self.video_frames[idx]
        
        frames = []
        for img_path in frame_paths:
            image = cv2.imread(img_path)
            if image is None:
                continue # Edge case handling, shouldn't normally happen
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
            
        if self.transform:
            if getattr(self.transform, 'is_sequence_transform', False):
                frames = self.transform(frames)
            else:
                frames = [self.transform(img) for img in frames]
            
        # Stack frames along a new sequence dimension
        # If transform converts to tensor [C, H, W], stack produces [Seq, C, H, W]
        # Depending on the model, we might want [C, Seq, H, W], we'll use [Seq, C, H, W] for LSTM
        if isinstance(frames[0], torch.Tensor):
            sequence_tensor = torch.stack(frames) 
        else:
            # If no transform or returned numpy arrays
            sequence_tensor = torch.tensor(np.array(frames)).float()
            # If shape is [Seq, H, W, C], convert to [Seq, C, H, W]
            if sequence_tensor.dim() == 4 and sequence_tensor.shape[-1] == 3:
                 sequence_tensor = sequence_tensor.permute(0, 3, 1, 2)

        label_tensor = torch.tensor(label).float()

        return sequence_tensor, label_tensor