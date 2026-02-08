#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ESP_I2S.h>

// ====== USER CONFIG ======
const char* WIFI_SSID = "gan1";
const char* WIFI_PASS = "18676180003";

// Local proxy (your Mac IP)
const char* WS_HOST = "192.168.8.134"; // TODO: change to your Mac IP
const uint16_t WS_PORT = 8765;
const char* WS_PATH = "/";

// XIAO ESP32S3 Sense PDM mic pins (fixed on-board)
// CLK = GPIO42, DATA = GPIO41
#define PIN_PDM_CLK  42
#define PIN_PDM_DATA 41

// Audio settings
static const int SAMPLE_RATE = 16000;
static const int BITS_PER_SAMPLE = 16;
static const int CHANNELS = 1;
static const int FRAME_MS = 40;
static const int SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS / 1000; // 640
static const int BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2;           // 1280

WebSocketsClient ws;
I2SClass I2S;
bool ws_connected = false;
static unsigned long last_wifi_attempt_ms = 0;
static unsigned long last_ws_attempt_ms = 0;
const unsigned long RECONNECT_INTERVAL_MS = 60000;
static unsigned long last_ws_state_log_ms = 0;
const unsigned long WS_STATE_LOG_INTERVAL_MS = 5000;
static unsigned long last_audio_stat_log_ms = 0;
const unsigned long AUDIO_STAT_LOG_INTERVAL_MS = 5000;
static size_t audio_bytes_sent = 0;
static size_t audio_frames_sent = 0;
static bool stream_active = false;

static void mic_setup() {
  I2S.setPinsPdmRx(PIN_PDM_CLK, PIN_PDM_DATA);
  bool ok = I2S.begin(
    I2S_MODE_PDM_RX,
    SAMPLE_RATE,
    I2S_DATA_BIT_WIDTH_16BIT,
    I2S_SLOT_MODE_MONO
  );
  if (!ok) {
    Serial.println("I2S mic init failed");
  } else {
    Serial.println("I2S mic init OK");
  }
}

static void wifi_connect() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  Serial.print("WiFi connected, IP: ");
  Serial.println(WiFi.localIP());
}

static void ws_setup() {
  ws.begin(WS_HOST, WS_PORT, WS_PATH);
  ws.setExtraHeaders("User-Agent: esp32");
  ws.setReconnectInterval(2000);
  ws.onEvent([](WStype_t type, uint8_t* payload, size_t length) {
    if (type == WStype_CONNECTED) {
      ws_connected = true;
      Serial.println("WS connected");
      return;
    }
    if (type == WStype_DISCONNECTED) {
      ws_connected = false;
      Serial.println("WS disconnected");
      return;
    }
    if (type == WStype_TEXT) {
      // Print ASR text results from proxy
      String msg((char*)payload, length);
      Serial.println(msg);
    }
    if (type == WStype_ERROR) {
      Serial.print("WS error: ");
      if (length > 0 && payload != nullptr) {
        Serial.write(payload, length);
      }
      Serial.println();
      return;
    }
  });
}

void setup() {
  Serial.begin(115200);
  delay(500);

  wifi_connect();
  ws_setup();
  mic_setup();
  Serial.println("Setup done");
  Serial.println("Send 'r' + Enter to start/stop chat stream.");
}

void loop() {
  unsigned long now = millis();
  // Serial control: 'r' toggles stream
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      continue;
    }
    if (c == 'r' || c == 'R') {
      stream_active = !stream_active;
      if (stream_active) {
        Serial.println("Stream ON");
        if (!ws_connected) {
          ws.disconnect();
          ws.begin(WS_HOST, WS_PORT, WS_PATH);
        }
      } else {
        Serial.println("Stream OFF");
        if (ws_connected) {
          ws.sendTXT("{\"type\":\"end\"}");
          ws.disconnect();
        }
      }
    }
  }
  if (WiFi.status() != WL_CONNECTED) {
    if (now - last_wifi_attempt_ms >= RECONNECT_INTERVAL_MS) {
      last_wifi_attempt_ms = now;
      Serial.println("WiFi reconnect attempt...");
      WiFi.disconnect(true);
      WiFi.begin(WIFI_SSID, WIFI_PASS);
    }
  } else {
    if (!ws_connected && now - last_ws_attempt_ms >= RECONNECT_INTERVAL_MS) {
      last_ws_attempt_ms = now;
      Serial.println("WS reconnect attempt...");
      ws.disconnect();
      ws.begin(WS_HOST, WS_PORT, WS_PATH);
    }
  }

  if (stream_active) {
    ws.loop();
  }

  if (stream_active && !ws_connected && now - last_ws_state_log_ms >= WS_STATE_LOG_INTERVAL_MS) {
    last_ws_state_log_ms = now;
    Serial.printf("WS status: connected=%d\n", ws_connected ? 1 : 0);
  }

  if (!stream_active) {
    return;
  }

  static int16_t frame[SAMPLES_PER_FRAME];
  int samples = 0;
  unsigned long start_ms = millis();

  while (samples < SAMPLES_PER_FRAME && (millis() - start_ms) < 200) {
    int sample = I2S.read();
    if (sample == 0 || sample == -1 || sample == 1) {
      continue;
    }
    frame[samples++] = (int16_t)sample;
  }

  if (samples == SAMPLES_PER_FRAME) {
    if (ws_connected) {
      const uint8_t* bytes = reinterpret_cast<const uint8_t*>(frame);
      ws.sendBIN(bytes, BYTES_PER_FRAME);
      audio_bytes_sent += BYTES_PER_FRAME;
      audio_frames_sent += 1;
    }
  }

  if (now - last_audio_stat_log_ms >= AUDIO_STAT_LOG_INTERVAL_MS) {
    last_audio_stat_log_ms = now;
    Serial.printf("Audio sent: frames=%u bytes=%u\n",
                  (unsigned int)audio_frames_sent,
                  (unsigned int)audio_bytes_sent);
  }
}
