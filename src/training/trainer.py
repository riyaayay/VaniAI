import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
import logging
import numpy as np
from pathlib import Path
from src.data.augmentations import AudioAugmentor

logger = logging.getLogger(__name__)

class IndicGuardTrainer:
    def __init__(self, model, config, train_loader, val_loader, device):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.train_cfg = config["training"]
        self.epochs = self.train_cfg.get("epochs", 40)
        self.accumulation_steps = self.train_cfg.get("accumulate_grad_batches", 1)
        
        self.augmentor = AudioAugmentor(config).to(device)
        
        # === SUPER CONVERGENCE SETUP ===
        # We use a higher max_lr because OneCycle handles the warmup/cooldown
        max_lr = float(self.train_cfg["optimizer"].get("learning_rate", 1e-3))
        wd = float(self.train_cfg["optimizer"].get("weight_decay", 0.01))
        
        self.optimizer = optim.AdamW(model.parameters(), lr=max_lr/10, weight_decay=wd)
        
        # OneCycleLR: The secret to fast, high-accuracy training
        steps_per_epoch = len(train_loader) // self.accumulation_steps
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,  # Spend 30% of time warming up
            div_factor=10,
            final_div_factor=1000
        )
        
        # Label Smoothing: The secret to low EER
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.scaler = GradScaler('cuda')
        self.best_eer = 1.0
        self.ckpt_dir = Path(config["paths"]["checkpoints"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def compute_eer(self, labels, scores):
        scores = np.nan_to_num(scores, nan=0.0)
        if len(np.unique(labels)) < 2: return 0.5 
        try:
            fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
            return brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        except: return 0.5

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch}", leave=False)
        self.optimizer.zero_grad()
        
        valid_batches = 0
        
        for batch_idx, batch in enumerate(pbar):
            waveforms = batch['waveform'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            with torch.no_grad():
                waveforms = self.augmentor(waveforms)
            
            with autocast('cuda'):
                output = self.model(waveforms)
                loss = self.criterion(output['logits'], labels)
                loss = loss / self.accumulation_steps
            
            if not torch.isfinite(loss):
                self.optimizer.zero_grad()
                continue
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step() # Step scheduler EVERY BATCH for OneCycle
                self.optimizer.zero_grad()
            
            loss_val = loss.item() * self.accumulation_steps
            running_loss += loss_val
            valid_batches += 1
            
            _, predicted = torch.max(output['logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'L': f"{loss_val:.3f}", 'A': f"{100*correct/total:.1f}%"})
            
        avg_loss = running_loss / max(valid_batches, 1)
        acc = 100 * correct / max(total, 1)
        return avg_loss, acc

    def validate(self):
        self.model.eval()
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                waveforms = batch['waveform'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                with autocast('cuda'):
                    output = self.model(waveforms)
                probs = output['probs'][:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(probs.cpu().float().numpy())
        
        return self.compute_eer(all_labels, all_scores)

    def train(self, num_epochs):
        logger.info(f"Training on {self.device} (Accumulating {self.accumulation_steps})...")
        
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Check EER every 2 epochs (Frequent checks for short deadlines)
            if epoch % 2 == 0 or epoch == num_epochs:
                eer = self.validate()
                logger.info(f"Ep {epoch}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%, EER={eer:.4%}")
                
                if eer < self.best_eer:
                    self.best_eer = eer
                    torch.save(self.model.state_dict(), self.ckpt_dir / "best_model.pth")
                    logger.info("  -> Saved Best")
            else:
                logger.info(f"Ep {epoch}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")