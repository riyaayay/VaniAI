"""
Spectral Residual Encoder (ResNet-Audio) - NAN SAFE
"""

import torch
import torch.nn as nn
import torchaudio.functional as F
import logging
import math

logger = logging.getLogger(__name__)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SpectralEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        audio_cfg = config["audio"]
        model_cfg = config["model"]["spectral_encoder"]
        
        self.n_fft = audio_cfg["n_fft"]
        self.hop_length = audio_cfg["hop_length"]
        self.n_mels = audio_cfg["n_mels"]
        self.sample_rate = audio_cfg["sample_rate"]
        
        mel_basis = F.melscale_fbanks(
            n_freqs=(self.n_fft // 2) + 1,
            f_min=0.0,
            f_max=self.sample_rate / 2.0,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            norm='slaney',
            mel_scale='htk',
        )
        self.register_buffer('mel_basis', mel_basis)
        window = torch.hann_window(self.n_fft)
        self.register_buffer('window', window)
        
        self.output_dim = model_cfg["output_dim"]
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(32, 64, stride=(2, 1))
        self.layer2 = self._make_layer(64, 128, stride=(2, 1))
        self.layer3 = self._make_layer(128, 256, stride=(2, 1))
        self.layer4 = self._make_layer(256, self.output_dim, stride=(2, 1))
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None)) 

    def _make_layer(self, in_c, out_c, stride):
        return ResBlock(in_c, out_c, stride)
    
    def _safe_stft(self, waveform):
        if not waveform.is_contiguous():
            waveform = waveform.contiguous()
        complex_spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        real = complex_spec.real
        imag = complex_spec.imag
        # Clamp inputs to pow() to prevent Infinity
        real = torch.clamp(real, min=-1e4, max=1e4)
        imag = torch.clamp(imag, min=-1e4, max=1e4)
        mag_spec = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-9)
        return mag_spec

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mag_spec = self._safe_stft(waveform)
            melspec = torch.matmul(self.mel_basis.transpose(0, 1), mag_spec)
            # Safe Log: Clamp minimum to prevent -inf
            melspec = torch.log10(torch.clamp(melspec, min=1e-5, max=1e5))
            x = melspec.unsqueeze(1)
            
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) 
        x = self.freq_pool(x)
        x = x.squeeze(2)
        x = x.transpose(1, 2)
        return x