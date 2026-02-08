# Voice AI PoC (Tencent Cloud ASR)

This is a minimal local proxy for Tencent Cloud real-time ASR. Your ESP32-S3 device streams 16 kHz / 16-bit / mono PCM over WebSocket to this proxy, which then forwards to Tencent Cloud and returns text results.

## 1) Install Python deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r /Users/mac/Desktop/pet/proxy/requirements.txt
```

## 2) Configure env

Create a `.env` file in `/Users/mac/Desktop/pet/proxy`:

```env
TENCENT_APPID=your_appid
TENCENT_SECRET_ID=your_secret_id
TENCENT_SECRET_KEY=your_secret_key

# Optional tuning
ENGINE_MODEL_TYPE=16k_zh
VOICE_FORMAT=1
NEED_VAD=1
VAD_SILENCE_TIME=800
FILTER_PUNC=1

LOCAL_WS_HOST=0.0.0.0
LOCAL_WS_PORT=8765
```

## 3) Run proxy

```bash
source .venv/bin/activate
python /Users/mac/Desktop/pet/proxy/server.py
```

## 4) Device streaming expectations

- PCM 16 kHz, 16-bit, mono
- Recommended chunk size: 40 ms (1280 bytes)
- Use a WebSocket client to send binary audio frames to:
  `ws://<your-mac-ip>:8765`
- When an utterance ends, optionally send a text message:
  `{ "type": "end" }`

The proxy prints Tencent ASR results to stdout and also forwards them back to the device over the same local WebSocket.
