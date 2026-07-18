---
title: VaniAI Voice Detection
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
app_file: app.py
pinned: false
---

# VaniAI (IndicGuard) — AI Voice Detection API

Multi-language AI vs. Human voice detection for Tamil, English, Hindi, Malayalam, and Telugu.

Built using a SincNet + WavLM ensemble, achieving **2.67% Equal Error Rate (EER)** at time of evaluation, benchmarked across varied noise conditions.

## Contributors

- **Riya Rathod** — [github.com/riyaayay](https://github.com/riyaayay)
- **M. Runeet Kumar** — [realruneett@gmail.com](mailto:realruneett@gmail.com)

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

## Notes

The publicly deployed Hugging Face Space may not currently reflect the 2.67% EER benchmark achieved during original development and evaluation (model performance has degraded post-deployment). The codebase and documented methodology in this repository remain accurate and reproducible.
