import torch
import torch.nn as nn
import timm

class SpatialBranch(nn.Module):
    """
    Spatial feature extraction using DeiT-Small (Data-efficient Image Transformer).
    
    DeiT-Small is chosen over vanilla ViT because it is specifically designed 
    to work effectively with smaller datasets through knowledge distillation,
    making it ideal for deepfake detection where large-scale labeled data 
    may be limited.
    
    Architecture: Patch Embedding (16x16) → 12 Transformer Blocks → CLS Token → FC Projection
    
    - Parameters: ~22M (vs 2.5M for MobileNetV3-Small)
    - Embedding dim: 384
    - Patch grid: 14x14 (224/16 = 14)
    - Input:  [B, S, 3, 224, 224] (batch of video sequences)
    - Output: spatial_out [B, feature_dim], sequence_features [B, S, feature_dim]
    """
    
    def __init__(self, feature_dim=256, pretrained=True):
        super(SpatialBranch, self).__init__()
        
        # Load DeiT-Small pretrained on ImageNet via knowledge distillation
        # num_classes=0 removes the classification head, returning raw 384-d CLS token
        self.backbone = timm.create_model(
            'deit_small_patch16_224',
            pretrained=pretrained,
            num_classes=0
        )
        
        self.embed_dim = 384   # DeiT-Small embedding dimension
        self.grid_size = 14    # 224 / 16 = 14 patches per spatial dimension
        
        # Project 384-d transformer embedding down to feature_dim
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, feature_dim),
            nn.GELU(),         # Standard activation for Transformer architectures
            nn.Dropout(p=0.2)
        )
        
    def forward(self, sequences):
        # sequences shape: [Batch_size, Sequence_length, Channels, Height, Width]
        b, s, c, h, w = sequences.size()
        
        # Process each frame independently through the Vision Transformer
        # Reshape to treat (batch * sequence) as a massive batch of single frames
        x = sequences.view(b * s, c, h, w)
        
        # Extract features through DeiT backbone
        # forward_features returns [B*S, N+1, D] where N=196 patches, D=384
        # This includes CLS token at position 0 + 196 patch tokens
        features = self.backbone.forward_features(x)
        
        # Use CLS token as the frame-level feature representation
        cls_token = features[:, 0]  # Shape: [B*S, 384]
        
        # Project to feature_dim
        x = self.fc(cls_token)  # Shape: [B*S, feature_dim]
        
        # Reshape back to sequence format
        x = x.view(b, s, -1)  # Shape: [B, S, feature_dim]
        
        # Aggregate spatial features over time (mean pooling)
        spatial_out = torch.mean(x, dim=1)  # Shape: [B, feature_dim]
        
        # Return both the per-frame features (for the Temporal branch) 
        # and the aggregated features (for final Fusion)
        return spatial_out, x
