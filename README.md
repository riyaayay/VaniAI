---
title: IndicGuard Voice Detection
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
app_file: app.py
pinned: false
---

# IndicGuard AI Voice Detection API

Multi-language AI vs Human voice detection for Tamil, English, Hindi, Malayalam, and Telugu.

## API Usage

### Endpoint
```
POST /api/voice-detection
```

### Headers
```
Content-Type: application/json
x-api-key: hackathon_secret_key
```

### Request Body
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_ENCODED_MP3_STRING"
}
```

### Example
```bash
curl -X POST "https://your-space.hf.space/api/voice-detection" \
  -H "Content-Type: application/json" \
  -H "x-api-key: hackathon_secret_key" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "YOUR_BASE64_AUDIO"
  }'
```

## Supported Languages
- Tamil
- English
- Hindi
- Malayalam
- Telugu