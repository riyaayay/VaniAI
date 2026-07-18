#!/usr/bin/env python3
"""
IndicGuard Training Script - FRESH START
Target: EER < 6% (Bank-Grade Security)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import torch
import yaml
import numpy as np
import random

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataloader import create_dataloaders
from src.models.indicguard_model import IndicGuardModel
from src.training.trainer import IndicGuardTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(f'training.log')]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    config_path = Path(args.config)
    with open(config_path) as f: config = yaml.safe_load(f)
    
    set_seed(config["project"].get("seed", 42))
    
    # Create directories
    for k in ["checkpoints", "logs", "cache"]:
        Path(config["paths"].get(k, f"./{k}")).mkdir(parents=True, exist_ok=True)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # TF32 Acceleration (Crucial for Speed on RTX 50-series)
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)} (TF32 Enabled)")
    else:
        sys.exit("CUDA required")
    
    logger.info("Loading Data...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    logger.info("=" * 70)
    logger.info("FINE-TUNING PHASE - Optimizing Best Model (Epoch 34)")
    logger.info("=" * 70)
    
    model = IndicGuardModel(config)
    
    # =========================================================
    # CRITICAL UPDATE: Load the Epoch 34 Brain (17.2% EER)
    # =========================================================
    # This file was saved automatically when you hit 17.2% EER.
    # It was NOT overwritten by the later 27% EER epochs.
    ckpt_path = Path("./checkpoints_final/best_model.pth")
    
    if ckpt_path.exists():
        logger.info(f"Loading weights from: {ckpt_path}")
        logger.info("Restoring state from Epoch 34 (17.2% EER)...")
        
        # Load weights
        state_dict = torch.load(ckpt_path)
        model.load_state_dict(state_dict)
        
        logger.info("✅ Success! Starting Fine-Tuning from peak performance.")
    else:
        logger.error(f"❌ ERROR: Checkpoint not found at {ckpt_path}")
        logger.error("Cannot fine-tune without the model file!")
        sys.exit(1) # Stop so we don't accidentally train a dummy model
    # =========================================================
    
    trainer = IndicGuardTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    
    logger.info("Starting Fine-Tuning Run...")
    logger.info(f"Target: Push EER from 17% -> <6%")
    logger.info(f"Fine-Tuning Epochs: {config['training'].get('epochs', 50)}")
    
    try:
        trainer.train(config["training"].get("epochs", 50))
    except KeyboardInterrupt:
        logger.info("Interrupted by User")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    main(parser.parse_args())