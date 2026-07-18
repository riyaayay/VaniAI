"""
Bidirectional Mamba (BiMamba) Backbone - DEPLOYMENT VERSION
Forces PyTorch fallback to match checkpoint architecture
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# FORCE FALLBACK MODE FOR CHECKPOINT COMPATIBILITY
# Your best_model.pth was trained with the PyTorch fallback implementation.
# The official mamba_ssm has different architecture (adds .mamba submodule) which breaks loading.
MAMBA_AVAILABLE = False
logger.info("🔧 Forced PyTorch fallback mode for checkpoint compatibility")


# ─────────────────────────────────────────────────────────────
# Stochastic Depth (Layer 8)
# ─────────────────────────────────────────────────────────────

def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.,
    training: bool = False
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) - Layer 8"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    output = x / keep_prob * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ─────────────────────────────────────────────────────────────
# Fallback PyTorch implementation (matches your checkpoint)
# ─────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    PyTorch-native Mamba approximation.
    Used when official Mamba-SSM is not available.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise convolution
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # State space parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.register_buffer('A', -A)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def ssm_step(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        delta: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplified SSM step with numerical stability."""
        A = self.A.unsqueeze(0).unsqueeze(0)
        delta = delta.unsqueeze(-1).clamp(max=10.0)  # Numerical stability

        # Discretization with numerical stability
        A_bar = torch.exp(delta * A)
        B = B.unsqueeze(1)
        B_bar = delta * B

        # State update
        x_expanded = x.unsqueeze(-1)
        h = A_bar * h + B_bar * x_expanded

        # Output
        C = C.unsqueeze(1)
        y = (C * h).sum(dim=-1)
        y = y + self.D * x

        return y, h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        residual = x

        # Normalize
        x = self.norm(x)

        # Project
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Convolution
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = F.silu(x)

        # SSM parameters
        x_ssm = self.x_proj(x)
        delta, B, C = torch.split(x_ssm, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(delta.squeeze(-1))

        # SSM scan
        h = torch.zeros(
            batch, self.d_inner, self.d_state,
            device=x.device, dtype=x.dtype
        )

        outputs = []
        for t in range(seq_len):
            y, h = self.ssm_step(x[:, t, :], h, delta[:, t, :], B[:, t, :], C[:, t, :])
            outputs.append(y)

        y = torch.stack(outputs, dim=1)

        # Gate
        z = F.silu(z)
        y = y * z

        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)

        return residual + y


# ─────────────────────────────────────────────────────────────
# Bidirectional Mamba Layer
# ─────────────────────────────────────────────────────────────

class BiMambaLayer(nn.Module):
    """
    Bidirectional Mamba Layer - PRODUCTION OPTIMIZED

    Processes sequences in both directions for global context.
    Uses official Mamba when available for 10x speedup.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.mamba_fwd = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.mamba_bwd = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        self.fusion = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)

        # Stochastic depth (Layer 8)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional processing with automatic gradient checkpointing.

        Args:
            x: Input (batch, seq_len, d_model)
        Returns:
            Output (batch, seq_len, d_model)
        """
        # Forward direction
        fwd = self.mamba_fwd(x)

        # Backward direction (flip sequence)
        bwd = self.mamba_bwd(x.flip(1)).flip(1)

        # Fuse both directions
        combined = torch.cat([fwd, bwd], dim=-1)
        out = self.fusion(combined)
        out = self.fusion_norm(out)

        # Stochastic depth residual
        return x + self.drop_path(out - x)


# ─────────────────────────────────────────────────────────────
# BiMamba Backbone
# ─────────────────────────────────────────────────────────────

class BiMambaBackbone(nn.Module):
    """
    Complete BiMamba Backbone - PRODUCTION READY

    Key optimizations:
    - Uses official Mamba CUDA kernels when available
    - Gradient checkpointing for memory efficiency
    - Optimized for 12GB VRAM
    """

    def __init__(self, config: dict):
        super().__init__()

        # Handle multiple config nesting patterns
        if "bimamba" in config:
            bimamba_config = config["bimamba"]
        elif "model" in config and "bimamba" in config["model"]:
            bimamba_config = config["model"]["bimamba"]
        elif "backbone" in config:
            bimamba_config = config["backbone"]
        else:
            bimamba_config = config

        self.input_dim    = bimamba_config["input_dim"]
        self.hidden_dim   = bimamba_config["hidden_dim"]
        self.num_layers   = bimamba_config["num_layers"]
        self.d_state      = bimamba_config.get("d_state", 16)
        self.d_conv       = bimamba_config.get("d_conv", 4)
        self.expand       = bimamba_config["expand_factor"]
        self.dropout      = bimamba_config["dropout"]
        self.stochastic_depth = bimamba_config["stochastic_depth_prob"]
        self.use_fast_path = bimamba_config.get("use_fast_path", True) and MAMBA_AVAILABLE

        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)

        # Stochastic depth schedule
        dpr = [
            x.item()
            for x in torch.linspace(0, self.stochastic_depth, self.num_layers)
        ]

        # Build layers
        self.layers = nn.ModuleList([
            BiMambaLayer(
                d_model=self.hidden_dim,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout,
                drop_path=dpr[i],
            )
            for i in range(self.num_layers)
        ])

        self.output_norm = nn.LayerNorm(self.hidden_dim)

        logger.info(f"BiMamba Backbone initialized:")
        logger.info(f"  Layers: {self.num_layers}")
        logger.info(f"  Hidden Dim: {self.hidden_dim}")
        logger.info(f"  Fast Path (CUDA): {self.use_fast_path}")
        logger.info(f"  Stochastic Depth: {self.stochastic_depth}")

    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Process temporal sequence with automatic gradient checkpointing.

        Args:
            x: Input (batch, seq_len, input_dim)
            return_sequence: If True, return full sequence; else return mean-pooled
        Returns:
            Output (batch, hidden_dim) or (batch, seq_len, hidden_dim)
        """
        # Handle 2D input (batch, features) → (batch, 1, features)
        squeezed = x.dim() == 2
        if squeezed:
            x = x.unsqueeze(1)

        x = self.input_proj(x)
        x = self.input_norm(x)

        # Process through BiMamba layers with gradient checkpointing
        for layer in self.layers:
            if self.training and hasattr(torch.utils.checkpoint, 'checkpoint'):
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.output_norm(x)

        if return_sequence and not squeezed:
            return x

        # Mean pooling over sequence dimension → (batch, hidden_dim)
        return x.mean(dim=1)