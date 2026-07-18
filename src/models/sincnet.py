"""
SincNet Learnable Filterbank
Stage 2 - Stream B: Signal-Level Feature Extraction

FIXED: Removed all complex FFT operations, using only real-valued convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def ensure_real_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is real-valued float32."""
    if torch.is_complex(tensor):
        tensor = tensor.abs()
    return tensor.float()


class SincConv(nn.Module):
    """
    Sinc-based Convolutional Layer
    
    Implements learnable band-pass filters using sinc functions.
    Key for capturing high-frequency artifacts from neural vocoders.
    
    FIXED: All operations are real-valued, no FFT/complex operations.
    
    Reference: "Speaker Recognition from Raw Waveform with SincNet"
    """
    
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 16000,
        min_low_hz: float = 50,
        min_band_hz: float = 50,
        stride: int = 1,
        padding: str = "same",
    ):
        super().__init__()
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.stride = stride
        
        # Initialize filterbank with mel-scale frequencies
        low_hz = 30
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        
        # Mel scale initialization
        mel_low = self._hz_to_mel(low_hz)
        mel_high = self._hz_to_mel(high_hz)
        mel_points = np.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = self._mel_to_hz(mel_points)
        
        # Learnable parameters for low and band frequencies
        self.low_hz_ = nn.Parameter(torch.tensor(hz_points[:-1], dtype=torch.float32).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.tensor(np.diff(hz_points), dtype=torch.float32).view(-1, 1))
        
        # Hamming window (real-valued)
        n_lin = torch.linspace(0, (kernel_size / 2) - 1, steps=int((kernel_size / 2)))
        self.register_buffer('window_', 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size))
        
        # Time points for sinc functions (real-valued)
        n = (self.kernel_size - 1) / 2.0
        self.register_buffer(
            'n_',
            2 * math.pi * torch.arange(-n, 0, dtype=torch.float32).view(1, -1) / sample_rate
        )
    
    @staticmethod
    def _hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    @staticmethod
    def _mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Apply sinc filterbank convolution.
        
        Args:
            waveforms: (batch, samples) or (batch, 1, samples) - MUST be real-valued
        
        Returns:
            Filtered output: (batch, out_channels, time) - real-valued
        """
        # CRITICAL: Ensure input is real-valued
        waveforms = ensure_real_tensor(waveforms)
        
        # Ensure 3D input
        if waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(1)
        
        # Compute filter parameters (all real-valued operations)
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2
        )
        band = (high - low)[:, 0]
        
        # Compute sinc filters (real-valued trigonometric functions)
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        
        # Band-pass filter using sinc function (real-valued)
        # sinc(x) = sin(x) / x for real x
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) 
            / (self.n_ + 1e-8)
        ) * self.window_.to(waveforms.device)
        
        # Create symmetric filter (real-valued)
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        
        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], 
            dim=1
        )
        
        # Normalize filters (real-valued)
        band_pass = band_pass / (2 * band[:, None] + 1e-8)
        
        # Ensure filters are real and finite
        band_pass = ensure_real_tensor(band_pass)
        band_pass = torch.where(
            torch.isfinite(band_pass), 
            band_pass, 
            torch.zeros_like(band_pass)
        )
        
        # Reshape for convolution: (out_channels, 1, kernel_size)
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        
        # Apply convolution (real-valued operation)
        output = F.conv1d(
            waveforms,
            filters.to(waveforms.device),
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=1,
        )
        
        # Final check: ensure output is real
        return ensure_real_tensor(output)


class SincNetEncoder(nn.Module):
    """
    Complete SincNet-based Signal Encoder
    
    Architecture:
    - SincConv layer (learnable filterbank)
    - Batch normalization (Layer 3: Anti-overfitting)
    - Convolutional layers for feature extraction
    - Dropout (Layer 1: Anti-overfitting)
    
    FIXED: All operations are real-valued, no FFT/complex numbers.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        sincnet_config = config["model"]["sincnet"]
        training_config = config["training"]
        
        self.num_filters = sincnet_config["num_filters"]
        self.kernel_size = sincnet_config["kernel_size"]
        self.sample_rate = sincnet_config["sample_rate"]
        self.output_dim = sincnet_config["output_dim"]
        self.dropout_rate = training_config["dropout_rate"]
        
        # Stage 1: Sinc filterbank (real-valued)
        self.sinc_conv = SincConv(
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            sample_rate=self.sample_rate,
            min_low_hz=sincnet_config["min_low_hz"],
            min_band_hz=sincnet_config["min_band_hz"],
            stride=sincnet_config["stride"],
        )
        
        # Batch normalization (Layer 3)
        self.bn1 = nn.BatchNorm1d(
            self.num_filters,
            momentum=training_config.get("batch_norm_momentum", 0.1),
            eps=training_config.get("batch_norm_eps", 1e-5),
        )
        
        # Stage 2: Convolutional processing (all real-valued)
        self.conv_blocks = nn.Sequential(
            # Conv block 1
            nn.Conv1d(self.num_filters, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),  # Layer 1
            
            # Conv block 2
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
            
            # Conv block 3
            nn.Conv1d(256, self.output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.output_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
        )
        
        # Final projection
        self.output_norm = nn.LayerNorm(self.output_dim)
        
        logger.info(f"SincNet Encoder initialized:")
        logger.info(f"  Filters: {self.num_filters}")
        logger.info(f"  Output Dim: {self.output_dim}")
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract signal-level features from waveform.
        
        Args:
            waveform: Raw audio (batch, samples) - MUST be real-valued
        
        Returns:
            Features: (batch, seq_len, output_dim) - real-valued
        """
        # CRITICAL: Ensure input is real
        waveform = ensure_real_tensor(waveform)
        
        # Apply sinc filterbank
        x = self.sinc_conv(waveform)  # (batch, num_filters, time)
        
        # Ensure still real after sinc conv
        x = ensure_real_tensor(x)
        
        # Batch normalization
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        
        # Apply convolutional blocks
        x = self.conv_blocks(x)  # (batch, output_dim, time')
        
        # Transpose for output: (batch, time, features)
        x = x.transpose(1, 2)
        
        # Apply layer normalization
        x = self.output_norm(x)
        
        return ensure_real_tensor(x)
    
    def match_length(
        self, 
        sincnet_features: torch.Tensor, 
        target_length: int
    ) -> torch.Tensor:
        """
        Match SincNet output length to WavLM output length.
        
        Args:
            sincnet_features: (batch, seq_len, features)
            target_length: Target sequence length
        
        Returns:
            Resampled features: (batch, target_length, features)
        """
        sincnet_features = ensure_real_tensor(sincnet_features)
        batch, seq_len, features = sincnet_features.shape
        
        if seq_len == target_length:
            return sincnet_features
        
        # Transpose for interpolation
        x = sincnet_features.transpose(1, 2)  # (batch, features, seq_len)
        
        # Interpolate to target length (real-valued operation)
        x = F.interpolate(
            x, 
            size=target_length, 
            mode='linear', 
            align_corners=False
        )
        
        # Transpose back
        return ensure_real_tensor(x.transpose(1, 2))  # (batch, target_length, features)