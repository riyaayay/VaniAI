"""
IndicGuard: Complete Deepfake Detection Model
WavLM-Free Architecture (Spectral + SincNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import logging

# Replaced WavLM with SpectralEncoder
from .spectral_encoder import SpectralEncoder
from .sincnet import SincNetEncoder
from .bimamba import BiMambaBackbone
from .liquid_layer import LiquidLayer
from .kan_classifier import KANClassifier

logger = logging.getLogger(__name__)

class IndicGuardModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Stream A: Spectral Encoder (Replaces WavLM)
        logger.info("Initializing Spectral Encoder (Stream A)...")
        self.spectral = SpectralEncoder(config)
        
        # Stream B: SincNet Encoder (Signal)
        logger.info("Initializing SincNet Encoder (Stream B)...")
        self.sincnet = SincNetEncoder(config)
        
        # Fusion Dimensions
        spectral_dim = config["model"]["spectral_encoder"]["output_dim"]
        sincnet_dim = config["model"]["sincnet"]["output_dim"]
        fused_dim = spectral_dim + sincnet_dim
        
        # Update BiMamba input
        config["model"]["bimamba"]["input_dim"] = fused_dim
        
        # Stage 3: BiMamba Backbone
        logger.info("Initializing BiMamba Backbone...")
        self.bimamba = BiMambaBackbone(config)
        
        # Stage 4: Liquid-KAN
        logger.info("Initializing Liquid-KAN Decision Engine...")
        self.liquid = LiquidLayer(config)
        self.kan = KANClassifier(config)
        
        # Projection for Liquid Layer
        self.bimamba_output_dim = config["model"]["bimamba"]["hidden_dim"]
        liquid_input = config["model"]["liquid"]["input_dim"]
        
        if self.bimamba_output_dim != liquid_input:
            self.bimamba_to_liquid = nn.Linear(self.bimamba_output_dim, liquid_input)
        else:
            self.bimamba_to_liquid = nn.Identity()
            
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Validate Input
        if torch.is_complex(waveform): waveform = waveform.abs()
        waveform = waveform.float()
        
        # Stream A: Spectral Features (Semantic/Texture)
        spectral_feat = self.spectral(waveform) # (B, Time_Spec, Dim_Spec)
        
        # Stream B: SincNet Features (Artifacts)
        sincnet_feat = self.sincnet(waveform)   # (B, Time_Sinc, Dim_Sinc)
        
        # Align Time Dimensions (Interpolate Spectral to match SincNet)
        target_len = sincnet_feat.shape[1]
        if spectral_feat.shape[1] != target_len:
            spectral_feat = spectral_feat.transpose(1, 2) # (B, Dim, Time)
            spectral_feat = F.interpolate(spectral_feat, size=target_len, mode='linear')
            spectral_feat = spectral_feat.transpose(1, 2) # (B, Time, Dim)
            
        # Fusion
        fused = torch.cat([spectral_feat, sincnet_feat], dim=-1)
        
        # Temporal Processing
        temporal_out = self.bimamba(fused)
        
        # Decision
        liquid_in = self.bimamba_to_liquid(temporal_out)
        liquid_out = self.liquid(liquid_in)
        logits = self.kan(liquid_out)
        
        probs = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probs': probs
        }