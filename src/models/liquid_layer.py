"""
Liquid Neural Network Layer
Stage 4a: Adaptive Time-Constant Processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class LiquidTimeConstant(nn.Module):
    """
    Liquid Time-Constant (LTC) Cell
    
    Implements continuous-time dynamics with adaptive time constants.
    The "liquid" behavior allows the network to adjust its reaction
    speed based on input volatility.
    
    Reference: "Liquid Time-constant Networks" (Hasani et al., 2020)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tau_min: float = 0.1,
        tau_max: float = 10.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        # Input-to-hidden mapping
        self.W_in = nn.Linear(input_dim, hidden_dim)
        
        # Hidden-to-hidden mapping
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        
        # Time constant modulation
        # Tau is computed as: tau = tau_min + (tau_max - tau_min) * sigmoid(tau_net)
        self.tau_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Output gate
        self.W_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization for stability
        self.ln_h = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        dt: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single LTC step.
        
        Args:
            x: Input (batch, input_dim)
            h: Previous hidden state (batch, hidden_dim)
            dt: Time step
        
        Returns:
            output: (batch, hidden_dim)
            new_h: (batch, hidden_dim)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if None
        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Compute adaptive time constant
        tau_input = torch.cat([x, h], dim=-1)
        tau_raw = self.tau_net(tau_input)
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(tau_raw)
        
        # Compute input contribution
        f_x = torch.tanh(self.W_in(x))
        
        # Compute recurrent contribution
        f_h = torch.tanh(self.W_h(h))
        
        # ODE update: dh/dt = (1/tau) * (-h + f(x, h))
        # Euler discretization: h_new = h + dt * (1/tau) * (-h + f_x + f_h)
        activation = f_x + f_h
        dh = (dt / tau) * (-h + activation)
        h_new = h + dh
        
        # Apply layer normalization for stability
        h_new = self.ln_h(h_new)
        
        # Output
        output = torch.tanh(self.W_out(h_new))
        
        return output, h_new


class LiquidLayer(nn.Module):
    """
    Complete Liquid Neural Network Layer
    
    Key Features:
    - Adaptive time constants based on input volatility
    - ODE-based dynamics for robustness to noise
    - Dropout for regularization (Layer 1)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        liquid_config = config["model"]["liquid"]
        
        self.input_dim = liquid_config["input_dim"]
        self.hidden_dim = liquid_config["hidden_dim"]
        self.tau_min = liquid_config["tau_min"]
        self.tau_max = liquid_config["tau_max"]
        self.dt = liquid_config["dt"]
        self.num_steps = liquid_config["num_steps"]
        self.dropout_rate = liquid_config["dropout"]
        
        # LTC cell
        self.ltc = LiquidTimeConstant(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            tau_min=self.tau_min,
            tau_max=self.tau_max,
        )
        
        # Input projection (if dimensions don't match)
        if self.input_dim != self.hidden_dim:
            self.input_proj = nn.Linear(self.input_dim, self.input_dim)
        else:
            self.input_proj = nn.Identity()
        
        # Dropout (Layer 1)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Output layer normalization
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        
        logger.info(f"Liquid Layer initialized:")
        logger.info(f"  Hidden Dim: {self.hidden_dim}")
        logger.info(f"  Tau Range: [{self.tau_min}, {self.tau_max}]")
        logger.info(f"  Num Steps: {self.num_steps}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through Liquid layer.
        
        The layer runs multiple ODE steps to "think" about the input,
        with the time constant adapting based on input complexity.
        
        Args:
            x: Input features (batch, input_dim)
        
        Returns:
            Output features (batch, hidden_dim)
        """
        # Project input
        x = self.input_proj(x)
        
        # Initialize hidden state
        h = None
        
        # Run multiple liquid steps
        for _ in range(self.num_steps):
            output, h = self.ltc(x, h, self.dt)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Normalize output
        output = self.output_norm(output)
        
        return output
    
    def get_time_constants(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the adaptive time constants for interpretability.
        
        Returns:
            Time constants (batch, hidden_dim)
        """
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        x = self.input_proj(x)
        
        tau_input = torch.cat([x, h], dim=-1)
        tau_raw = self.ltc.tau_net(tau_input)
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(tau_raw)
        
        return tau