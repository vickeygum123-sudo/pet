# Arduino Voice Streamer (XIAO ESP32S3 Sense)

This sketch streams 16 kHz / 16-bit / mono PCM audio over WebSocket to a local proxy.

## Setup

1) Install Arduino-ESP32 (ESP32 core) and the `WebSocketsClient` library.
2) Open `/Users/mac/Desktop/pet/arduino/voice_streamer/voice_streamer.ino`
3) Fill in:
   - `WIFI_SSID`, `WIFI_PASS`
   - `WS_HOST` with your Mac IP
   - `PIN_PDM_CLK` and `PIN_PDM_DATA` are fixed on XIAO ESP32S3 Sense (CLK=GPIO42, DATA=GPIO41)
4) Select the correct board and port in Arduino-IDE.
5) Upload and open Serial Monitor at 115200.

ASR results should appear in Serial Monitor.
