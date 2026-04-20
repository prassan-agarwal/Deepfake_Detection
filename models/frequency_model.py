import math
import torch
import torch.nn as nn

class FrequencyBranch(nn.Module):
    """
    Frequency-domain analysis branch using 2D DFT via matrix multiplication.
    
    Previous limitation: torch.fft.fft2 is not supported by ONNX, so the
    exported model output zeros instead of real frequency features.
    
    Fix: Implement 2D DFT as matrix multiplication (W @ x @ W), which uses
    only matmul/add/sqrt/log — all fully ONNX-compatible operations.
    The DFT matrices are pre-computed once and stored as model buffers.
    
    Mathematically identical to torch.fft.fft2, just uses O(N^2) matrix ops
    instead of O(N log N) FFT algorithm. For 224x224 images, this adds ~10ms
    per inference — negligible for video-level detection.
    """
    
    def __init__(self, feature_dim=128, img_size=224):
        super(FrequencyBranch, self).__init__()
        
        self.img_size = img_size
        
        # Pre-compute DFT basis matrices (not trainable, exported with ONNX)
        # DFT matrix: W[k,n] = exp(-j * 2pi * k * n / N) = cos(...) - j*sin(...)
        N = img_size
        indices = torch.arange(N, dtype=torch.float32)
        angles = 2.0 * math.pi * indices.unsqueeze(0) * indices.unsqueeze(1) / N  # [N, N]
        self.register_buffer('dft_cos', torch.cos(angles))  # Real part of DFT matrix
        self.register_buffer('dft_sin', torch.sin(angles))  # Imaginary part of DFT matrix
        
        # A tiny custom CNN to process the 2D FFT magnitude spectrum
        # Using minimal channels to save VRAM
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def _compute_fft_magnitude(self, x):
        """
        Compute 2D FFT magnitude spectrum via DFT matrix multiplication.
        Fully ONNX-compatible — uses only matmul, add, sqrt, log, and roll.
        
        Math: For real input x, the 2D DFT is X = W @ x @ W where W = C - jS.
              X_real = C@x@C - S@x@S
              X_imag = -(C@x@S + S@x@C)
              Magnitude = sqrt(X_real^2 + X_imag^2)
        
        Args:
            x: [B, C, H, W] real-valued image tensor
        Returns:
            [B, C, H, W] log-scaled, centered magnitude spectrum
        """
        # Force FP32 to prevent overflow during mixed precision (FP16) training.
        # DFT matrix multiplications produce large intermediate values that
        # exceed FP16's max representable value (~65504), causing NaN.
        original_dtype = x.dtype
        x = x.float()
        
        C = self.dft_cos  # [N, N]
        S = self.dft_sin  # [N, N]
        
        # Step 1: Left multiply (column-wise DFT)
        Cx = torch.matmul(C, x)   # [B, Ch, N, N]
        Sx = torch.matmul(S, x)   # [B, Ch, N, N]
        
        # Step 2: Right multiply (row-wise DFT) — W is symmetric so W^T = W
        X_real = torch.matmul(Cx, C) - torch.matmul(Sx, S)
        X_imag = -(torch.matmul(Cx, S) + torch.matmul(Sx, C))
        
        # Magnitude spectrum
        magnitude = torch.sqrt(X_real ** 2 + X_imag ** 2 + 1e-8)
        
        # Log scaling to compress dynamic range
        magnitude_spectrum = torch.log(magnitude + 1e-8)
        
        # fftshift: center the zero-frequency component
        half = self.img_size // 2
        magnitude_spectrum = torch.roll(magnitude_spectrum, shifts=(half, half), dims=(-2, -1))
        
        return magnitude_spectrum
        
    def forward(self, sequences):
        # sequences shape: [Batch_size, Sequence_length, Channels, Height, Width]
        b, s, c, h, w = sequences.size()
        
        # Select the middle frame of the sequence for frequency analysis
        mid_idx = s // 2
        center_frames = sequences[:, mid_idx, :, :, :]  # Shape: [B, C, H, W]
        
        # Disable mixed precision for DFT computation to prevent FP16 overflow.
        # autocast in train.py would otherwise cast matmul ops to FP16,
        # causing values to exceed FP16 max (~65504) and produce NaN.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            magnitude_spectrum = self._compute_fft_magnitude(center_frames.float())
        
        # Pass through our small CNN (this can safely use FP16 via autocast)
        out = self.cnn(magnitude_spectrum)
        out = torch.flatten(out, 1)
        freq_out = self.fc(out)  # Shape: [Batch_size, feature_dim]
        
        return freq_out
