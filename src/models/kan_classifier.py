"""
Kolmogorov-Arnold Network (KAN) Classifier
Stage 4b: Interpretable Decision Engine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math
import logging

logger = logging.getLogger(__name__)


class BSplineBasis(nn.Module):
    """
    B-Spline Basis Functions for KAN
    
    B-splines provide smooth, localized basis functions
    that enable interpretable learned transformations.
    """
    
    def __init__(
        self,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        
        # Number of basis functions
        self.num_basis = grid_size + spline_order
        
        # Create uniform grid (extended for boundary conditions)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer('grid', grid)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate B-spline basis functions at points x.
        
        Args:
            x: Input points (batch, features)
        
        Returns:
            Basis values (batch, features, num_basis)
        """
        x = x.unsqueeze(-1)  # (batch, features, 1)
        grid = self.grid  # (num_knots,)
        
        # Recursive B-spline computation
        # B_i,0(x) = 1 if grid[i] <= x < grid[i+1] else 0
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        
        for k in range(1, self.spline_order + 1):
            # B_i,k(x) = (x - grid[i]) / (grid[i+k] - grid[i]) * B_i,k-1(x)
            #          + (grid[i+k+1] - x) / (grid[i+k+1] - grid[i+1]) * B_i+1,k-1(x)
            
            left_num = x - grid[:-k-1].unsqueeze(0).unsqueeze(0)
            left_den = grid[k:-1] - grid[:-k-1] + 1e-8
            left_term = left_num / left_den.unsqueeze(0).unsqueeze(0) * bases[..., :-1]
            
            right_num = grid[k+1:].unsqueeze(0).unsqueeze(0) - x
            right_den = grid[k+1:] - grid[1:-k] + 1e-8
            right_term = right_num / right_den.unsqueeze(0).unsqueeze(0) * bases[..., 1:]
            
            bases = left_term + right_term
        
        return bases


class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer
    
    Unlike MLPs which use fixed activations, KAN learns
    univariate functions on the edges using B-splines.
    
    Reference: "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # B-spline basis
        self.basis = BSplineBasis(
            grid_size=grid_size,
            spline_order=spline_order,
        )
        
        # Number of basis functions
        num_basis = grid_size + spline_order
        
        # Learnable spline coefficients for each edge
        # Shape: (out_features, in_features, num_basis)
        self.coef = nn.Parameter(
            torch.randn(out_features, in_features, num_basis) * 0.1
        )
        
        # Residual (silu) connection for stability
        self.residual_weight = nn.Parameter(torch.ones(out_features, in_features) * 0.1)
        
        # Scale factor
        self.scale = nn.Parameter(torch.ones(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN layer.
        
        Args:
            x: Input (batch, in_features)
        
        Returns:
            Output (batch, out_features)
        """
        batch_size = x.size(0)
        
        # Normalize input to grid range
        x_norm = torch.tanh(x)  # Map to [-1, 1]
        
        # Evaluate B-spline basis
        basis = self.basis(x_norm)  # (batch, in_features, num_basis)
        
        # Compute spline output
        # For each output neuron, sum over input neurons
        # output[j] = sum_i sum_k coef[j,i,k] * basis[i,k]
        spline_out = torch.einsum('bik,jik->bj', basis, self.coef)
        
        # Add residual (silu) connection
        residual = torch.einsum('bi,ji->bj', F.silu(x), self.residual_weight)
        
        # Combine with scale
        output = self.scale * (spline_out + residual)
        
        return output
    
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get importance of each input feature.
        
        Returns:
            Feature importance (in_features,)
        """
        # Use coefficient magnitudes as proxy for importance
        importance = self.coef.abs().mean(dim=(0, 2))  # (in_features,)
        importance = importance / importance.sum()
        return importance


class KANClassifier(nn.Module):
    """
    Complete KAN Classifier for Deepfake Detection
    
    Key Features:
    - Learnable B-spline functions for interpretability
    - Feature importance extraction for explanations
    - Dropout for regularization (Layer 1)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        kan_config = config["model"]["kan"]
        
        self.input_dim = kan_config["input_dim"]
        self.hidden_dim = kan_config["hidden_dim"]
        self.output_dim = kan_config["output_dim"]
        self.grid_size = kan_config["grid_size"]
        self.spline_order = kan_config["spline_order"]
        self.dropout_rate = kan_config["dropout"]
        
        # KAN layers
        self.kan1 = KANLayer(
            in_features=self.input_dim,
            out_features=self.hidden_dim,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
        )
        
        self.kan2 = KANLayer(
            in_features=self.hidden_dim,
            out_features=self.output_dim,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
        )
        
        # Dropout (Layer 1)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        
        logger.info(f"KAN Classifier initialized:")
        logger.info(f"  Input: {self.input_dim} -> Hidden: {self.hidden_dim} -> Output: {self.output_dim}")
        logger.info(f"  Grid Size: {self.grid_size}, Spline Order: {self.spline_order}")
    
    def forward(
        self, 
        x: torch.Tensor,
        return_logits: bool = True,
    ) -> torch.Tensor:
        """
        Classify input.
        
        Args:
            x: Input features (batch, input_dim)
            return_logits: If True, return raw logits; else probabilities
        
        Returns:
            Logits or probabilities (batch, output_dim)
        """
        # First KAN layer
        x = self.kan1(x)
        x = self.ln1(x)
        x = self.dropout(x)
        
        # Second KAN layer
        logits = self.kan2(x)
        
        if return_logits:
            return logits
        else:
            return F.softmax(logits, dim=-1)
    
    def get_explanation(
        self, 
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate explanation for classification.
        
        Returns:
            Dictionary with:
            - 'feature_importance': Input feature importance
            - 'hidden_importance': Hidden layer importance
            - 'prediction': Predicted class
            - 'confidence': Confidence score
        """
        # Get predictions
        with torch.no_grad():
            logits = self.forward(x, return_logits=True)
            probs = F.softmax(logits, dim=-1)
            
            if class_idx is None:
                class_idx = logits.argmax(dim=-1)
            
            confidence = probs.gather(1, class_idx.unsqueeze(-1)).squeeze(-1)
        
        # Get feature importance
        input_importance = self.kan1.get_feature_importance(x)
        hidden_importance = self.kan2.get_feature_importance(
            self.ln1(self.kan1(x))
        )
        
        return {
            'feature_importance': input_importance,
            'hidden_importance': hidden_importance,
            'prediction': class_idx,
            'confidence': confidence,
        }