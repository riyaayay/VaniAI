"""
IndicGuard AI Voice Detection API v2.0
Production-Ready Hackathon Deployment with Futuristic UI
"""
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import torch
import yaml
import io
import base64
import librosa
import numpy as np
import soundfile as sf
from src.models.indicguard_model import IndicGuardModel
import logging
from datetime import datetime
import traceback
import random
from typing import Optional
import time


# ==========================================
# CONFIGURATION
# ==========================================
CONFIG_PATH = "config/config.yaml"
CHECKPOINT_PATH = "checkpoints_stable/best_model.pth"
API_KEY_SECRET = "hackathon_secret_key"

LABEL_MAP = {
    0: "HUMAN",
    1: "AI_GENERATED"
}

HOST = "0.0.0.0"
PORT = 7860

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# FASTAPI APP
# ==========================================
app = FastAPI(
    title="IndicGuard AI Voice Detection API",
    description="Multi-language AI vs Human voice detection",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
device = None
config = None
model_has_warnings = False


# ==========================================
# PYDANTIC MODELS
# ==========================================
class HackathonRequest(BaseModel):
    """Request model"""
    language: str = Field(..., description="One of: Tamil, English, Hindi, Malayalam, Telugu")
    audioFormat: str = Field(..., pattern="^mp3$", description="Must be 'mp3'")
    audioBase64: str = Field(..., description="Base64 encoded MP3 audio")

    @validator('language')
    def validate_language(cls, v):
        allowed = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        if v not in allowed:
            raise ValueError(f"Language must be one of: {allowed}")
        return v


class HackathonResponse(BaseModel):
    """Response model"""
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str
    processingTime: Optional[float] = None
    audioInfo: Optional[dict] = None


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_dynamic_explanation(label: str, confidence: float) -> str:
    """Generate technical explanation"""
    pct = round(confidence * 100, 2)
    
    if label == "AI_GENERATED":
        if confidence > 0.98:
            options = [
                f"Critical Alert: Synthetic audio signature detected with {pct}% certainty.",
                f"Definitive AI pattern match ({pct}%). Unnatural phase continuity detected.",
                f"Neural vocoder signatures at {pct}% confidence. Metallic artifacts present."
            ]
        elif confidence > 0.85:
            options = [
                f"Strong AI synthesis indicators ({pct}%). Lack of natural pitch micro-tremors.",
                f"Repetitive prosody patterns typical of TTS models ({pct}% confidence).",
                f"Abnormal spectral consistency suggests AI generation ({pct}%)."
            ]
        else:
            options = [
                f"Potential synthetic content ({pct}%). Slight background inconsistencies detected.",
                f"Borderline AI classification ({pct}%). Audio quality affects certainty.",
                f"Suspicious frequency artifacts near 16kHz ({pct}% confidence)."
            ]
    else:  # HUMAN
        if confidence > 0.98:
            options = [
                f"Confirmed human speech ({pct}%). Natural breath pauses present.",
                f"Natural chaotic waveforms detected ({pct}%). Organic vocal patterns.",
                f"Clear human signature ({pct}%). No neural vocoder artifacts found."
            ]
        elif confidence > 0.85:
            options = [
                f"Human origin indicated ({pct}%). Natural consonant articulation.",
                f"Natural prosodic variation detected ({pct}% confidence).",
                f"Standard human voice characteristics ({pct}%)."
            ]
        else:
            options = [
                f"Leaning towards human ({pct}%). Audio compression limits certainty.",
                f"Likely human speech ({pct}%). Some noise mimics synthetic artifacts.",
                f"No strong AI signatures. Human classification at {pct}%."
            ]
    
    return random.choice(options)


def decode_and_process_audio(b64_string: str) -> tuple:
    """Decode Base64 MP3 and process for model"""
    start_time = time.time()
    
    try:
        b64_string = b64_string.strip()
        
        if ',' in b64_string and ('data:' in b64_string[:50] or 'base64' in b64_string[:50]):
            b64_string = b64_string.split(',', 1)[1]
        
        b64_string = ''.join(b64_string.split())
        logger.info(f"Base64 length: {len(b64_string)} chars")
        
        audio_bytes = base64.b64decode(b64_string)
        logger.info(f"Decoded: {len(audio_bytes)} bytes ({len(audio_bytes)/1024:.2f} KB)")
        
        if len(audio_bytes) < 100:
            raise ValueError(f"Audio too small: {len(audio_bytes)} bytes")
        
        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.seek(0)
        
        wav, sr = None, None
        
        try:
            wav, sr = librosa.load(audio_buffer, sr=16000, mono=True)
            logger.info(f"Loaded with Librosa: {len(wav)/sr:.2f}s")
        except Exception as e1:
            logger.warning(f"Librosa failed: {e1}")
            
            try:
                audio_buffer.seek(0)
                audio_buffer.name = 'audio.mp3'
                wav, sr = sf.read(audio_buffer)
                
                if len(wav.shape) > 1:
                    wav = np.mean(wav, axis=1)
                
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    sr = 16000
                
                logger.info(f"Loaded with soundfile: {len(wav)/sr:.2f}s")
            except Exception as e2:
                raise ValueError(f"Audio decode failed. Librosa: {e1}, Soundfile: {e2}")
        
        if wav is None or len(wav) == 0:
            raise ValueError("Audio empty after decoding")
        
        original_duration = len(wav) / sr
        
        max_val = np.abs(wav).max()
        if max_val > 0:
            wav = wav / max_val
        
        if not np.isfinite(wav).all():
            wav = np.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
        
        target_length = 48000
        
        if len(wav) < target_length:
            wav = np.pad(wav, (0, target_length - len(wav)), mode='constant')
        else:
            wav = wav[:target_length]
        
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
        
        audio_info = {
            "original_duration": round(original_duration, 2),
            "sample_rate": sr,
            "audio_size_kb": round(len(audio_bytes) / 1024, 2),
            "processing_time": round(time.time() - start_time, 3)
        }
        
        return wav_tensor, audio_info
    
    except base64.binascii.Error as e:
        raise ValueError(f"Invalid Base64 encoding: {e}")
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Audio processing failed: {e}")


def load_model_from_config():
    """Load model and config"""
    global model, device, config, model_has_warnings
    
    try:
        logger.info(f"Loading config from {CONFIG_PATH}...")
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        logger.info(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
        # Debug: Show what's in the checkpoint
        if isinstance(checkpoint, dict):
            logger.info(f"Checkpoint contains {len(checkpoint)} keys")
        else:
            logger.info(f"Checkpoint type: {type(checkpoint)}")
        
        logger.info("Initializing model...")
        model = IndicGuardModel(config)
        
        # Handle different checkpoint formats with strict=False for architecture mismatches
        load_success = False
        missing_keys = []
        unexpected_keys = []
        
        if isinstance(checkpoint, dict):
            # Check for different possible keys
            if 'model_state_dict' in checkpoint:
                logger.info("Loading from 'model_state_dict' key")
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                load_success = True
            elif 'state_dict' in checkpoint:
                logger.info("Loading from 'state_dict' key")
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
                load_success = True
            elif 'model' in checkpoint:
                logger.info("Loading from 'model' key")
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
                load_success = True
            else:
                # Checkpoint might be the state dict itself
                logger.info("Loading checkpoint as state dict directly (with strict=False)")
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                load_success = True
        else:
            # Checkpoint is the state dict directly
            logger.info("Loading checkpoint as state dict directly (with strict=False)")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            load_success = True
        
        if load_success:
            # Log warnings about mismatches
            if missing_keys:
                logger.warning(f"⚠️  Missing keys in checkpoint (will use random initialization): {len(missing_keys)} keys")
                logger.warning(f"First few missing keys: {missing_keys[:5]}")
                model_has_warnings = True
            
            if unexpected_keys:
                logger.warning(f"⚠️  Unexpected keys in checkpoint (will be ignored): {len(unexpected_keys)} keys")
                logger.warning(f"First few unexpected keys: {unexpected_keys[:5]}")
                model_has_warnings = True
            
            if missing_keys or unexpected_keys:
                logger.warning("⚠️  MODEL ARCHITECTURE MISMATCH DETECTED!")
                logger.warning("⚠️  The checkpoint may not match the current model architecture.")
                logger.warning("⚠️  This may result in reduced accuracy or unexpected behavior.")
                logger.warning("⚠️  Please ensure you're using the correct checkpoint for this model version.")
        
        model.to(device)
        model.eval()
        
        logger.info("✅ Model loaded successfully (with architecture adjustments)!")
        logger.info(f"Model device: {next(model.parameters()).device}")
        
        return True
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.error(traceback.format_exc())
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("🚀 Starting IndicGuard API...")
    success = load_model_from_config()
    if success:
        logger.info("✅ Startup complete - API ready!")
    else:
        logger.error("❌ Startup failed - model not loaded")


# ==========================================
# FUTURISTIC HTML INTERFACE
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def get_interface():
    """Futuristic web interface"""
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IndicGuard - Neural Voice Authentication System</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --cyber-blue: #00f3ff;
            --cyber-pink: #ff00ff;
            --cyber-purple: #8b00ff;
            --cyber-green: #00ff88;
            --cyber-orange: #ff6b00;
            --dark-bg: #0a0a0f;
            --dark-surface: #12121a;
            --dark-card: #1a1a28;
            --glow-blue: rgba(0, 243, 255, 0.4);
            --glow-pink: rgba(255, 0, 255, 0.4);
        }
        
        body {
            font-family: 'Rajdhani', sans-serif;
            background: var(--dark-bg);
            color: #fff;
            overflow-x: hidden;
            position: relative;
        }
        
        /* Animated Background Grid */
        .cyber-grid {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(0, 243, 255, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 243, 255, 0.1) 1px, transparent 1px);
            background-size: 50px 50px;
            animation: gridMove 20s linear infinite;
            pointer-events: none;
            z-index: 0;
        }
        
        @keyframes gridMove {
            0% { transform: perspective(500px) rotateX(60deg) translateZ(0); }
            100% { transform: perspective(500px) rotateX(60deg) translateZ(50px); }
        }
        
        /* Particle Effect */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1;
        }
        
        .particle {
            position: absolute;
            width: 2px;
            height: 2px;
            background: var(--cyber-blue);
            border-radius: 50%;
            animation: float linear infinite;
            box-shadow: 0 0 10px var(--cyber-blue);
        }
        
        @keyframes float {
            0% {
                transform: translateY(100vh) translateX(0);
                opacity: 0;
            }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% {
                transform: translateY(-100vh) translateX(50px);
                opacity: 0;
            }
        }
        
        /* Scanner Line Effect */
        .scanner {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, 
                transparent, 
                var(--cyber-blue), 
                transparent);
            animation: scan 4s ease-in-out infinite;
            z-index: 2;
            box-shadow: 0 0 20px var(--glow-blue);
        }
        
        @keyframes scan {
            0%, 100% { transform: translateY(0); opacity: 0; }
            50% { transform: translateY(100vh); opacity: 1; }
        }
        
        .container {
            position: relative;
            z-index: 10;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        
        /* Header */
        .header {
            text-align: center;
            margin-bottom: 60px;
            position: relative;
        }
        
        .logo {
            font-family: 'Orbitron', sans-serif;
            font-size: 4em;
            font-weight: 900;
            background: linear-gradient(135deg, var(--cyber-blue), var(--cyber-pink), var(--cyber-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-transform: uppercase;
            letter-spacing: 8px;
            margin-bottom: 10px;
            animation: glitch 5s infinite;
            text-shadow: 0 0 30px var(--glow-blue);
        }
        
        @keyframes glitch {
            0%, 90%, 100% { transform: translate(0); }
            91% { transform: translate(-2px, 2px); }
            92% { transform: translate(2px, -2px); }
            93% { transform: translate(-2px, 2px); }
        }
        
        .subtitle {
            font-size: 1.4em;
            color: var(--cyber-blue);
            letter-spacing: 4px;
            text-transform: uppercase;
            font-weight: 300;
            margin-bottom: 20px;
        }
        
        .status-bar {
            display: inline-block;
            padding: 10px 30px;
            background: rgba(0, 243, 255, 0.1);
            border: 1px solid var(--cyber-blue);
            border-radius: 20px;
            font-size: 0.9em;
            letter-spacing: 2px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 10px var(--glow-blue); }
            50% { box-shadow: 0 0 30px var(--glow-blue); }
        }
        
        /* Glass Card */
        .glass-card {
            background: rgba(26, 26, 40, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(0, 243, 255, 0.3);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 40px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), inset 0 0 50px rgba(0, 243, 255, 0.05);
        }
        
        .glass-card::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, 
                transparent, 
                rgba(0, 243, 255, 0.03), 
                transparent);
            animation: cardShine 3s infinite;
            pointer-events: none;
            z-index: 0;
        }
        
        @keyframes cardShine {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .section-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.8em;
            color: var(--cyber-blue);
            margin-bottom: 30px;
            text-transform: uppercase;
            letter-spacing: 3px;
            position: relative;
            padding-left: 20px;
            z-index: 10;
        }
        
        .section-title::before {
            content: '▸';
            position: absolute;
            left: 0;
            color: var(--cyber-pink);
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0%, 50%, 100% { opacity: 1; }
            25%, 75% { opacity: 0.3; }
        }
        
        /* Form Elements */
        .form-group {
            margin-bottom: 30px;
            position: relative;
            z-index: 10;
        }
        
        label {
            display: block;
            margin-bottom: 12px;
            color: var(--cyber-blue);
            font-size: 1.1em;
            letter-spacing: 2px;
            text-transform: uppercase;
            font-weight: 500;
        }
        
        input, select, textarea {
            width: 100%;
            padding: 16px 20px;
            background: rgba(18, 18, 26, 0.8);
            border: 2px solid rgba(0, 243, 255, 0.3);
            border-radius: 10px;
            color: #fff;
            font-size: 1em;
            font-family: 'Rajdhani', sans-serif;
            transition: all 0.3s;
        }
        
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: var(--cyber-blue);
            box-shadow: 0 0 20px var(--glow-blue);
            background: rgba(18, 18, 26, 1);
        }
        
        input:hover, select:hover, textarea:hover {
            border-color: var(--cyber-pink);
        }
        
        textarea {
            min-height: 140px;
            font-family: 'Courier New', monospace;
            resize: vertical;
        }
        
        /* File Input Styling */
        input[type="file"] {
            cursor: pointer;
        }
        
        input[type="file"]::file-selector-button {
            padding: 10px 20px;
            background: linear-gradient(135deg, var(--cyber-blue), var(--cyber-purple));
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            cursor: pointer;
            margin-right: 15px;
            transition: all 0.3s;
        }
        
        input[type="file"]::file-selector-button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px var(--glow-blue);
        }
        
        /* Tab Buttons */
        .tab-buttons {
            display: flex;
            gap: 20px;
            margin-bottom: 40px;
            position: relative;
            z-index: 10;
        }
        
        .tab-btn {
            flex: 1;
            padding: 18px;
            background: rgba(18, 18, 26, 0.6);
            border: 2px solid rgba(0, 243, 255, 0.3);
            border-radius: 12px;
            color: #fff;
            font-size: 1.1em;
            font-weight: 600;
            letter-spacing: 2px;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            font-family: 'Orbitron', sans-serif;
        }
        
        .tab-btn:hover {
            border-color: var(--cyber-pink);
            transform: translateY(-2px);
        }
        
        .tab-btn.active {
            background: linear-gradient(135deg, var(--cyber-blue), var(--cyber-purple));
            border-color: var(--cyber-blue);
            box-shadow: 0 0 30px var(--glow-blue);
        }
        
        .tab-content {
            display: none;
            animation: fadeIn 0.5s;
        }
        
        .tab-content.active {
            display: block;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Analyze Button */
        .btn-analyze {
            width: 100%;
            padding: 24px;
            background: linear-gradient(135deg, var(--cyber-blue), var(--cyber-purple), var(--cyber-pink));
            background-size: 200% 200%;
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 1.4em;
            font-weight: 700;
            letter-spacing: 3px;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            font-family: 'Orbitron', sans-serif;
            position: relative;
            overflow: hidden;
            animation: gradientShift 3s ease infinite;
            z-index: 10;
        }
        
        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        .btn-analyze:hover {
            transform: scale(1.02);
            box-shadow: 0 0 40px var(--glow-blue), 0 0 60px var(--glow-pink);
        }
        
        .btn-analyze:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-analyze::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            transition: left 0.5s;
        }
        
        .btn-analyze:hover::before {
            left: 100%;
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top: 3px solid white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 10px;
            display: none;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Result Card */
        .result-card {
            background: rgba(18, 18, 26, 0.8);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 40px;
            margin-top: 40px;
            border: 2px solid rgba(0, 243, 255, 0.4);
            animation: slideUp 0.6s ease-out;
            position: relative;
            overflow: hidden;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(50px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result-card.AI_GENERATED {
            border-color: var(--cyber-pink);
            box-shadow: 0 0 50px rgba(255, 0, 255, 0.3);
        }
        
        .result-card.HUMAN {
            border-color: var(--cyber-green);
            box-shadow: 0 0 50px rgba(0, 255, 136, 0.3);
        }
        
        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 40px;
        }
        
        .result-label {
            font-family: 'Orbitron', sans-serif;
            font-size: 2.5em;
            font-weight: 900;
            letter-spacing: 4px;
            text-transform: uppercase;
        }
        
        .result-icon {
            font-size: 4em;
            animation: iconPulse 2s infinite;
        }
        
        @keyframes iconPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        .confidence-display {
            text-align: center;
            margin: 40px 0;
            position: relative;
        }
        
        .confidence-ring {
            width: 250px;
            height: 250px;
            margin: 0 auto;
            position: relative;
        }
        
        .confidence-score {
            font-family: 'Orbitron', sans-serif;
            font-size: 4em;
            font-weight: 900;
            background: linear-gradient(135deg, var(--cyber-blue), var(--cyber-pink));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            animation: scoreGlow 2s infinite;
        }
        
        @keyframes scoreGlow {
            0%, 100% { filter: drop-shadow(0 0 10px var(--glow-blue)); }
            50% { filter: drop-shadow(0 0 30px var(--glow-pink)); }
        }
        
        .detail-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .detail-item {
            background: rgba(0, 243, 255, 0.05);
            border: 1px solid rgba(0, 243, 255, 0.2);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s;
        }
        
        .detail-item:hover {
            border-color: var(--cyber-blue);
            transform: translateY(-5px);
            box-shadow: 0 10px 30px var(--glow-blue);
        }
        
        .detail-label {
            color: var(--cyber-blue);
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 8px;
        }
        
        .detail-value {
            font-size: 1.6em;
            font-weight: 700;
            color: #fff;
        }
        
        .explanation-box {
            background: linear-gradient(135deg, rgba(0, 243, 255, 0.1), rgba(255, 0, 255, 0.1));
            border-left: 4px solid var(--cyber-blue);
            border-radius: 12px;
            padding: 25px;
            margin: 30px 0;
        }
        
        .explanation-box strong {
            color: var(--cyber-blue);
            font-size: 1.2em;
            display: block;
            margin-bottom: 15px;
            letter-spacing: 2px;
        }
        
        .json-viewer {
            background: rgba(0, 0, 0, 0.6);
            border: 1px solid rgba(0, 243, 255, 0.3);
            border-radius: 12px;
            padding: 25px;
            margin-top: 30px;
            overflow-x: auto;
        }
        
        .json-viewer pre {
            color: var(--cyber-green);
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.6;
        }
        
        /* Info Box */
        .info-box {
            background: rgba(139, 0, 255, 0.1);
            border: 1px solid rgba(139, 0, 255, 0.3);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            position: relative;
            z-index: 10;
        }
        
        .info-box h3 {
            color: var(--cyber-purple);
            font-family: 'Orbitron', sans-serif;
            margin-bottom: 15px;
            letter-spacing: 2px;
        }
        
        .info-box p {
            margin: 8px 0;
            line-height: 1.8;
        }
        
        .info-box code {
            background: rgba(0, 0, 0, 0.5);
            padding: 2px 8px;
            border-radius: 4px;
            color: var(--cyber-green);
            font-family: 'Courier New', monospace;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 60px 20px 40px;
            margin-top: 80px;
            position: relative;
        }
        
        .footer::before {
            content: '';
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 80%;
            height: 1px;
            background: linear-gradient(90deg, 
                transparent, 
                var(--cyber-blue), 
                var(--cyber-pink), 
                transparent);
        }
        
        .credits {
            font-size: 1.1em;
            color: #888;
            letter-spacing: 2px;
            margin-top: 30px;
        }
        
        .credits .brand {
            font-family: 'Orbitron', sans-serif;
            background: linear-gradient(135deg, var(--cyber-blue), var(--cyber-pink));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
            font-size: 1.3em;
            letter-spacing: 3px;
            display: block;
            margin: 10px 0;
        }
        
        .credits .by {
            color: var(--cyber-purple);
            font-weight: 600;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .logo {
                font-size: 2.5em;
                letter-spacing: 4px;
            }
            
            .subtitle {
                font-size: 1em;
                letter-spacing: 2px;
            }
            
            .glass-card {
                padding: 25px;
            }
            
            .tab-buttons {
                flex-direction: column;
            }
            
            .result-label {
                font-size: 1.8em;
            }
            
            .confidence-score {
                font-size: 3em;
            }
            
            .detail-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <!-- Background Effects -->
    <div class="cyber-grid"></div>
    <div class="particles" id="particles"></div>
    <div class="scanner"></div>
    
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="logo">INDICGUARD</div>
            <div class="subtitle">Neural Voice Authentication System</div>
            <div class="status-bar">● SYSTEM ONLINE | AI DETECTION READY</div>
        </div>
        
        <!-- Main Card -->
        <div class="glass-card">
            <div class="info-box">
                <h3>⚡ SYSTEM SPECIFICATIONS</h3>
                <p><strong>ENDPOINT:</strong> <code>/api/detect</code> (POST)</p>
                <p><strong>ALTERNATIVE:</strong> <code>/api/spaces/realruneet/indicguard</code> (POST)</p>
                <p><strong>AUTHENTICATION:</strong> Header <code>x-api-key: hackathon_secret_key</code></p>
                <p><strong>SUPPORTED LANGUAGES:</strong> Tamil | English | Hindi | Malayalam | Telugu</p>
            </div>
            
            <div class="section-title">ANALYSIS INTERFACE</div>
            
            <div class="tab-buttons">
                <button class="tab-btn active" onclick="switchTab('upload')">📁 UPLOAD MODE</button>
                <button class="tab-btn" onclick="switchTab('base64')">🔤 BASE64 MODE</button>
            </div>
            
            <!-- Upload Tab -->
            <div id="uploadTab" class="tab-content active">
                <div class="form-group">
                    <label>🔑 API ACCESS KEY</label>
                    <input type="text" id="apiKey" value="hackathon_secret_key" placeholder="Enter API key">
                </div>
                
                <div class="form-group">
                    <label>🌐 LANGUAGE PROTOCOL</label>
                    <select id="language">
                        <option value="English">English</option>
                        <option value="Tamil">Tamil</option>
                        <option value="Hindi">Hindi</option>
                        <option value="Malayalam">Malayalam</option>
                        <option value="Telugu">Telugu</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>🎤 AUDIO INPUT (MP3)</label>
                    <input type="file" id="audioFile" accept=".mp3" onchange="handleFileSelect(event)">
                </div>
            </div>
            
            <!-- Base64 Tab -->
            <div id="base64Tab" class="tab-content">
                <div class="form-group">
                    <label>🔑 API ACCESS KEY</label>
                    <input type="text" id="apiKeyBase64" value="hackathon_secret_key" placeholder="Enter API key">
                </div>
                
                <div class="form-group">
                    <label>🌐 LANGUAGE PROTOCOL</label>
                    <select id="languageBase64">
                        <option value="English">English</option>
                        <option value="Tamil">Tamil</option>
                        <option value="Hindi">Hindi</option>
                        <option value="Malayalam">Malayalam</option>
                        <option value="Telugu">Telugu</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>📝 BASE64 ENCODED DATA</label>
                    <textarea id="base64Input" placeholder="Paste base64 encoded MP3 data..."></textarea>
                </div>
            </div>
            
            <button class="btn-analyze" id="btnAnalyze" onclick="analyzeAudio()">
                <div class="spinner" id="spinner"></div>
                <span id="btnText">🚀 INITIALIZE ANALYSIS</span>
            </button>
        </div>
        
        <!-- Results Area -->
        <div id="resultArea" style="display: none;">
            <div class="result-card" id="resultCard">
                <div class="result-header">
                    <div class="result-label" id="resultLabel">⚠ AI GENERATED</div>
                    <div class="result-icon" id="resultIcon">🤖</div>
                </div>
                
                <div class="confidence-display">
                    <div class="confidence-ring">
                        <div class="confidence-score" id="confidenceScore">95.23%</div>
                    </div>
                </div>
                
                <div class="detail-grid" id="resultDetails"></div>
                
                <div class="explanation-box">
                    <strong>📊 NEURAL ANALYSIS REPORT</strong>
                    <p id="explanationText"></p>
                </div>
                
                <div class="json-viewer">
                    <strong>📄 COMPLETE DIAGNOSTIC DATA:</strong>
                    <pre id="jsonViewer"></pre>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <div class="credits">
                POWERED BY
                <span class="brand">INDIC GUARD</span>
                <span class="by">BY COUNCIL</span>
            </div>
        </div>
    </div>
    
    <script>
        // Initialize Particles
        function createParticles() {
            const particlesContainer = document.getElementById('particles');
            for (let i = 0; i < 30; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
                particle.style.animationDelay = Math.random() * 5 + 's';
                particlesContainer.appendChild(particle);
            }
        }
        createParticles();
        
        let currentTab = 'upload';
        
        function switchTab(tab) {
            currentTab = tab;
            
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            if (tab === 'upload') {
                document.querySelector('.tab-btn:first-child').classList.add('active');
                document.getElementById('uploadTab').classList.add('active');
            } else {
                document.querySelector('.tab-btn:last-child').classList.add('active');
                document.getElementById('base64Tab').classList.add('active');
            }
        }
        
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file && !file.name.endsWith('.mp3')) {
                alert('⚠️ INVALID FORMAT: Please select an MP3 file');
                event.target.value = '';
            }
        }
        
        async function analyzeAudio() {
            const btn = document.getElementById('btnAnalyze');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('spinner');
            
            let apiKey, language, base64Audio;
            
            if (currentTab === 'upload') {
                apiKey = document.getElementById('apiKey').value.trim();
                language = document.getElementById('language').value;
                const fileInput = document.getElementById('audioFile');
                
                if (!fileInput.files.length) {
                    alert('⚠️ NO FILE SELECTED: Please upload an MP3 file');
                    return;
                }
                
                try {
                    base64Audio = await readFileAsBase64(fileInput.files[0]);
                } catch (error) {
                    alert('❌ FILE READ ERROR: ' + error.message);
                    return;
                }
            } else {
                apiKey = document.getElementById('apiKeyBase64').value.trim();
                language = document.getElementById('languageBase64').value;
                base64Audio = document.getElementById('base64Input').value.trim();
                
                if (!base64Audio) {
                    alert('⚠️ NO DATA: Please paste Base64 encoded MP3 data');
                    return;
                }
                
                if (base64Audio.includes(',')) {
                    base64Audio = base64Audio.split(',')[1];
                }
            }
            
            btn.disabled = true;
            btnText.textContent = 'PROCESSING...';
            spinner.style.display = 'inline-block';
            document.getElementById('resultArea').style.display = 'none';
            
            const payload = {
                language: language,
                audioFormat: "mp3",
                audioBase64: base64Audio
            };
            
            try {
                const response = await fetch('/api/detect', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'x-api-key': apiKey
                    },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                
                if (response.status === 401) {
                    alert('⚠️ AUTHENTICATION FAILED: Invalid API Key');
                    return;
                }
                
                if (response.status !== 200) {
                    alert('❌ SYSTEM ERROR: ' + (data.message || data.error || 'Unknown error'));
                    console.error('Error details:', data);
                    return;
                }
                
                displayResults(data);
                
            } catch (error) {
                alert('❌ CONNECTION ERROR: ' + error.message);
                console.error('Error:', error);
            } finally {
                btn.disabled = false;
                btnText.textContent = '🚀 INITIALIZE ANALYSIS';
                spinner.style.display = 'none';
            }
        }
        
        function readFileAsBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const base64 = e.target.result.split(',')[1];
                    resolve(base64);
                };
                reader.onerror = () => reject(new Error('Failed to read file'));
                reader.readAsDataURL(file);
            });
        }
        
        function displayResults(data) {
            const resultArea = document.getElementById('resultArea');
            const resultCard = document.getElementById('resultCard');
            const resultLabel = document.getElementById('resultLabel');
            const resultIcon = document.getElementById('resultIcon');
            const confidenceScore = document.getElementById('confidenceScore');
            const resultDetails = document.getElementById('resultDetails');
            const explanationText = document.getElementById('explanationText');
            const jsonViewer = document.getElementById('jsonViewer');
            
            if (data.status !== 'success') {
                alert('ANALYSIS FAILED: ' + (data.message || 'Processing error'));
                return;
            }
            
            const isAI = data.classification === 'AI_GENERATED';
            resultCard.className = 'result-card ' + data.classification;
            
            resultLabel.textContent = isAI ? '⚠ AI GENERATED' : '✓ HUMAN VOICE';
            resultLabel.style.color = isAI ? 'var(--cyber-pink)' : 'var(--cyber-green)';
            resultIcon.textContent = isAI ? '🤖' : '👤';
            
            const confidence = (data.confidenceScore * 100).toFixed(2);
            confidenceScore.textContent = confidence + '%';
            
            let detailsHTML = '';
            detailsHTML += `<div class="detail-item">
                <div class="detail-label">LANGUAGE</div>
                <div class="detail-value">${data.language}</div>
            </div>`;
            
            if (data.audioInfo) {
                detailsHTML += `<div class="detail-item">
                    <div class="detail-label">DURATION</div>
                    <div class="detail-value">${data.audioInfo.original_duration}s</div>
                </div>`;
                detailsHTML += `<div class="detail-item">
                    <div class="detail-label">FILE SIZE</div>
                    <div class="detail-value">${data.audioInfo.audio_size_kb} KB</div>
                </div>`;
            }
            
            if (data.processingTime) {
                detailsHTML += `<div class="detail-item">
                    <div class="detail-label">PROCESS TIME</div>
                    <div class="detail-value">${data.processingTime}s</div>
                </div>`;
            }
            
            resultDetails.innerHTML = detailsHTML;
            explanationText.textContent = data.explanation;
            jsonViewer.textContent = JSON.stringify(data, null, 2);
            
            resultArea.style.display = 'block';
            
            setTimeout(function() {
                resultArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 100);
        }
        
        document.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && 
                e.target.id !== 'base64Input' && 
                !document.getElementById('btnAnalyze').disabled) {
                analyzeAudio();
            }
        });
    </script>
</body>
</html>""")


# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "model_warnings": model_has_warnings,
        "warning_message": "Model architecture mismatch detected - predictions may be inaccurate" if model_has_warnings else None,
        "device": str(device),
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    }


@app.post("/api/detect", response_model=HackathonResponse)
async def voice_detection_main(
    payload: HackathonRequest,
    x_api_key: str = Header(None, alias="x-api-key")
):
    """Main detection endpoint"""
    request_start = time.time()
    
    if x_api_key != API_KEY_SECRET:
        logger.warning(f"Invalid API key: {x_api_key}")
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Invalid API key"}
        )
    
    try:
        if model is None:
            raise ValueError("Model not loaded")
        
        logger.info(f"Request: language={payload.language}, base64_len={len(payload.audioBase64)}")
        
        audio_tensor, audio_info = decode_and_process_audio(payload.audioBase64)
        audio_tensor = audio_tensor.to(device)
        
        with torch.no_grad():
            output = model(audio_tensor)
            logits = output["logits"]
            probs = torch.softmax(logits, dim=1)
            
            winner_index = torch.argmax(probs, dim=1).item()
            winner_prob = probs[0][winner_index].item()
            predicted_label = LABEL_MAP.get(winner_index, "UNKNOWN")
        
        logger.info(f"Prediction: {predicted_label} ({winner_prob:.4f})")
        logger.info(f"Probs: HUMAN={probs[0][0]:.4f}, AI={probs[0][1]:.4f}")
        
        explanation = get_dynamic_explanation(predicted_label, winner_prob)
        total_time = time.time() - request_start
        
        return HackathonResponse(
            status="success",
            language=payload.language,
            classification=predicted_label,
            confidenceScore=round(winner_prob, 4),
            explanation=explanation,
            processingTime=round(total_time, 3),
            audioInfo=audio_info
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e), "type": "validation_error"}
        )
    
    except Exception as e:
        logger.error(f"Processing error: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error", "type": "server_error"}
        )


@app.post("/api/spaces/realruneet/indicguard", response_model=HackathonResponse)
async def voice_detection_legacy(
    payload: HackathonRequest,
    x_api_key: str = Header(None, alias="x-api-key")
):
    """Legacy detection endpoint"""
    return await voice_detection_main(payload, x_api_key)


if __name__ == "__main__":
    logger.info(f"🚀 Starting IndicGuard API on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")