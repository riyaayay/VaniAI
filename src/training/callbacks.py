"""
Training Callbacks for IndicGuard
Includes Early Stopping (Layer 5) and Model Checkpointing.
"""

import os
from pathlib import Path
from typing import Optional, Dict
import logging
import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early Stopping Callback (Layer 5)
    
    Stops training when a monitored metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = 'min',
        verbose: bool = True,
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' (minimize or maximize metric)
            verbose: Whether to log messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == 'min':
            self.compare = lambda current, best: current < best - min_delta
            self.best_score = float('inf')
        else:
            self.compare = lambda current, best: current > best + min_delta
            self.best_score = float('-inf')
    
    def __call__(self, current_score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current metric value
            epoch: Current epoch number
        
        Returns:
            True if training should stop
        """
        if self.compare(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            self.best_epoch = epoch
            if self.verbose:
                logger.info(
                    f"EarlyStopping: New best score {current_score:.6f} at epoch {epoch}"
                )
            return False
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs"
                )
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(
                        f"EarlyStopping: Stopping training. Best score {self.best_score:.6f} "
                        f"at epoch {self.best_epoch}"
                    )
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.early_stop = False
        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')


class ModelCheckpoint:
    """
    Model Checkpointing Callback
    
    Saves model checkpoints based on monitored metric.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_top_k: int = 3,
        save_last: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_top_k: Number of best checkpoints to keep
            save_last: Whether to save last checkpoint
            verbose: Whether to log messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose
        
        # Track best checkpoints: (score, epoch, path)
        self.best_checkpoints = []
        
        if mode == 'min':
            self.compare = lambda a, b: a < b
        else:
            self.compare = lambda a, b: a > b
    
    def __call__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        metrics: Dict[str, float],
    ) -> Optional[str]:
        """
        Save checkpoint if warranted.
        
        Returns:
            Path to saved checkpoint, or None
        """
        current_score = metrics.get(self.monitor, None)
        
        if current_score is None:
            logger.warning(f"Metric {self.monitor} not found in metrics")
            return None
        
        # Check if this is a top-k checkpoint
        should_save = False
        
        if len(self.best_checkpoints) < self.save_top_k:
            should_save = True
        else:
            # Compare with worst of current best
            worst_best = sorted(
                self.best_checkpoints, 
                key=lambda x: x[0], 
                reverse=(self.mode == 'min')
            )[-1]
            
            if self.compare(current_score, worst_best[0]):
                should_save = True
                # Remove worst checkpoint
                old_path = worst_best[2]
                if os.path.exists(old_path):
                    os.remove(old_path)
                self.best_checkpoints.remove(worst_best)
        
        saved_path = None
        
        if should_save:
            # Save checkpoint
            checkpoint_name = f"checkpoint_epoch{epoch}_{self.monitor}={current_score:.4f}.pt"
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }
            
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint, checkpoint_path)
            
            self.best_checkpoints.append((current_score, epoch, str(checkpoint_path)))
            
            if self.verbose:
                logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            saved_path = str(checkpoint_path)
            
            # Update best model symlink
            best_path = self.checkpoint_dir / "best_model.pt"
            
            # Find actual best
            if self.mode == 'min':
                best = min(self.best_checkpoints, key=lambda x: x[0])
            else:
                best = max(self.best_checkpoints, key=lambda x: x[0])
            
            # Copy best checkpoint
            import shutil
            shutil.copy(best[2], best_path)
        
        # Save last checkpoint
        if self.save_last:
            last_path = self.checkpoint_dir / "last_model.pt"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint, last_path)
        
        return saved_path
    
    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict:
        """Load best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        
        if not best_path.exists():
            raise FileNotFoundError(f"No best checkpoint found at {best_path}")
        
        checkpoint = torch.load(best_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint