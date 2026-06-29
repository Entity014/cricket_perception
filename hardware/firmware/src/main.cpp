#include <Arduino.h>
#include <HTTPClient.h>
#include <SD.h>
#include <SPI.h>
#include <WebServer.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include "driver/i2s.h"

namespace
{
  constexpr int MIC_WS_PIN = 42;
  constexpr int MIC_SCK_PIN = 41;
  constexpr int MIC_SD_PIN = 40;
  constexpr int SD_CS_PIN = 10;

  constexpr uint32_t SAMPLE_RATE = 16000;
  constexpr uint16_t WAV_BITS_PER_SAMPLE = 16;
  constexpr uint16_t WAV_CHANNELS = 1;
  constexpr size_t I2S_SAMPLES = 512;
  constexpr int16_t MIC_GAIN = 4;

  constexpr bool IS_MAIN_NODE = true; // Set to false on each sub node
  constexpr char MAIN_AP_SSID[] = "Cricket-Audio";
  constexpr char MAIN_AP_PASSWORD[] = "12345678";
  constexpr char MAIN_NODE_IP[] = "192.168.4.1";
  const char *REMOTE_NODE_IPS[] = {
      "192.168.4.2",
      "192.168.4.3",
  };

  WebServer server(80);
  File recordingFile;
  String recordingPath;
  bool isRecording = false;
  bool sdReady = false;
  bool i2sReady = false;
  uint32_t wavDataBytes = 0;
  String recordingLabel = "default";

  int32_t i2sRaw[I2S_SAMPLES];
  int16_t pcmBuffer[I2S_SAMPLES];

  void writeLE16(File &file, uint16_t value)
  {
    file.write(value & 0xff);
    file.write((value >> 8) & 0xff);
  }

  void writeLE32(File &file, uint32_t value)
  {
    file.write(value & 0xff);
    file.write((value >> 8) & 0xff);
    file.write((value >> 16) & 0xff);
    file.write((value >> 24) & 0xff);
  }

  void writeWavHeader(File &file, uint32_t dataBytes)
  {
    const uint32_t byteRate = SAMPLE_RATE * WAV_CHANNELS * WAV_BITS_PER_SAMPLE / 8;
    const uint16_t blockAlign = WAV_CHANNELS * WAV_BITS_PER_SAMPLE / 8;

    file.seek(0);
    file.write(reinterpret_cast<const uint8_t *>("RIFF"), 4);
    writeLE32(file, 36 + dataBytes);
    file.write(reinterpret_cast<const uint8_t *>("WAVE"), 4);
    file.write(reinterpret_cast<const uint8_t *>("fmt "), 4);
    writeLE32(file, 16);
    writeLE16(file, 1);
    writeLE16(file, WAV_CHANNELS);
    writeLE32(file, SAMPLE_RATE);
    writeLE32(file, byteRate);
    writeLE16(file, blockAlign);
    writeLE16(file, WAV_BITS_PER_SAMPLE);
    file.write(reinterpret_cast<const uint8_t *>("data"), 4);
    writeLE32(file, dataBytes);
  }

  String makeRecordingPath()
  {
    uint32_t index = 1;
    String path;

    do
    {
      path = "/rec_" + String(index) + ".wav";
      index++;
    } while (SD.exists(path) && index < 100000);

    return path;
  }

  bool setupI2S()
  {
    const i2s_config_t i2sConfig = {
        .mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 6,
        .dma_buf_len = 512,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0,
    };

    const i2s_pin_config_t pinConfig = {
        .mck_io_num = I2S_PIN_NO_CHANGE,
        .bck_io_num = MIC_SCK_PIN,
        .ws_io_num = MIC_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = MIC_SD_PIN,
    };

    esp_err_t result = i2s_driver_install(I2S_NUM_0, &i2sConfig, 0, nullptr);
    if (result != ESP_OK)
    {
      Serial.printf("I2S driver install failed: %d\n", result);
      return false;
    }

    result = i2s_set_pin(I2S_NUM_0, &pinConfig);
    if (result != ESP_OK)
    {
      Serial.printf("I2S pin config failed: %d\n", result);
      return false;
    }

    i2s_zero_dma_buffer(I2S_NUM_0);
    return true;
  }

  bool broadcastSyncCommand(const String &path, const String &label)
  {
    const String localIp = WiFi.localIP().toString();
    bool anyReached = false;

    for (size_t i = 0; i < (sizeof(REMOTE_NODE_IPS) / sizeof(REMOTE_NODE_IPS[0])); ++i)
    {
      const String remoteIp = String(REMOTE_NODE_IPS[i]);
      if (remoteIp == localIp)
      {
        continue;
      }

      WiFiClient client;
      HTTPClient http;
      String url = "http://" + remoteIp + path + "?label=" + label;
      if (!http.begin(client, url))
      {
        Serial.printf("Unable to start sync request to %s\n", remoteIp.c_str());
        continue;
      }

      http.setTimeout(2000);
      int httpCode = http.GET();
      http.end();

      if (httpCode >= 200 && httpCode < 300)
      {
        anyReached = true;
        Serial.printf("Synced %s to %s\n", path.c_str(), remoteIp.c_str());
      }
      else
      {
        Serial.printf("Sync request to %s failed: %d\n", remoteIp.c_str(), httpCode);
      }
    }

    return anyReached;
  }

  String sanitizeLabel(const String &label)
  {
    String sanitized = label;
    sanitized.trim();
    sanitized.replace(' ', '_');
    sanitized.replace('/', '_');
    sanitized.replace('\\', '_');
    sanitized.replace(':', '_');
    sanitized.replace('&', 'a');
    if (sanitized.length() == 0)
    {
      sanitized = "default";
    }
    return sanitized;
  }

  String jsonStatus()
  {
    String json = "{";
    json += "\"recording\":";
    json += isRecording ? "true" : "false";
    json += ",\"file\":\"";
    json += recordingPath;
    json += "\",\"bytes\":";
    json += wavDataBytes;
    json += ",\"sdReady\":";
    json += sdReady ? "true" : "false";
    json += ",\"i2sReady\":";
    json += i2sReady ? "true" : "false";
    json += ",\"ip\":\"";
    json += WiFi.localIP().toString();
    json += "\",\"label\":\"";
    json += recordingLabel;
    json += "\"}";
    return json;
  }

  void sendJsonStatus()
  {
    server.send(200, "application/json", jsonStatus());
  }

  bool startRecording(const String &label = "")
  {
    if (isRecording)
    {
      return true;
    }
    if (!sdReady || !i2sReady)
    {
      Serial.println("Cannot start recording: SD or I2S is not ready");
      return false;
    }

    recordingLabel = sanitizeLabel(label.length() > 0 ? label : recordingLabel);
    recordingPath = makeRecordingPath();
    recordingFile = SD.open(recordingPath, FILE_WRITE);
    if (!recordingFile)
    {
      Serial.println("Failed to create recording file");
      return false;
    }

    String metadataPath = recordingPath.substring(0, recordingPath.lastIndexOf('.')) + ".txt";
    File metadataFile = SD.open(metadataPath, FILE_WRITE);
    if (metadataFile)
    {
      metadataFile.println(String("label=") + recordingLabel);
      metadataFile.println(String("node=") + (IS_MAIN_NODE ? "main" : "sub"));
      metadataFile.close();
    }

    wavDataBytes = 0;
    writeWavHeader(recordingFile, wavDataBytes);
    i2s_zero_dma_buffer(I2S_NUM_0);
    isRecording = true;

    Serial.print("Recording started: ");
    Serial.print(recordingPath);
    Serial.print(" label=");
    Serial.println(recordingLabel);
    return true;
  }

  void stopRecording()
  {
    if (!isRecording)
    {
      return;
    }

    isRecording = false;
    writeWavHeader(recordingFile, wavDataBytes);
    recordingFile.close();

    Serial.print("Recording saved: ");
    Serial.print(recordingPath);
    Serial.print(" (");
    Serial.print(wavDataBytes);
    Serial.println(" data bytes)");
  }

  void handleRemoteStart()
  {
    String label = server.hasArg("label") ? server.arg("label") : String("default");
    startRecording(label);
    sendJsonStatus();
  }

  void handleRemoteStop()
  {
    stopRecording();
    sendJsonStatus();
  }

  void handleStart()
  {
    String label = server.hasArg("label") ? server.arg("label") : String("default");
    if (!startRecording(label))
    {
      server.send(500, "application/json", "{\"error\":\"failed to start recording\"}");
      return;
    }

    broadcastSyncCommand("/sync/start", label);
    sendJsonStatus();
  }

  void handleStop()
  {
    stopRecording();
    broadcastSyncCommand("/sync/stop", recordingLabel);
    sendJsonStatus();
  }

  void handleDownload()
  {
    if (!server.hasArg("file"))
    {
      server.send(400, "text/plain", "Missing file parameter");
      return;
    }

    String path = server.arg("file");
    if (!path.startsWith("/"))
    {
      path = "/" + path;
    }

    if (!SD.exists(path))
    {
      server.send(404, "text/plain", "File not found");
      return;
    }

    File file = SD.open(path, FILE_READ);
    server.streamFile(file, "audio/wav");
    file.close();
  }

  void handleFiles()
  {
    if (!sdReady)
    {
      server.send(200, "application/json", "[]");
      return;
    }

    String json = "[";
    File root = SD.open("/");
    if (!root)
    {
      server.send(200, "application/json", "[]");
      return;
    }

    File file = root.openNextFile();
    bool first = true;

    while (file)
    {
      String name = file.name();
      if (!file.isDirectory() && name.endsWith(".wav"))
      {
        if (!first)
        {
          json += ",";
        }
        json += "{\"name\":\"";
        json += name.startsWith("/") ? name : "/" + name;
        json += "\",\"size\":";
        json += file.size();
        json += "}";
        first = false;
      }
      file.close();
      file = root.openNextFile();
    }
    root.close();

    json += "]";
    server.send(200, "application/json", json);
  }

  void handleIndex()
  {
    static const char page[] PROGMEM = R"HTML(
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cricket Audio Recorder</title>
  <style>
    :root { color-scheme: light dark; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; min-height: 100vh; background: #f3f5f7; color: #18202a; }
    main { width: min(760px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 20px; }
    h1 { font-size: clamp(1.55rem, 5vw, 2.25rem); margin: 0; letter-spacing: 0; }
    .status { font-weight: 700; color: #516171; }
    .panel { background: #fff; border: 1px solid #d8e0e8; border-radius: 8px; padding: 18px; box-shadow: 0 8px 26px rgba(24, 32, 42, 0.08); }
    .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 16px 0; }
    button { min-height: 48px; border: 0; border-radius: 8px; font-size: 1rem; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: 0.45; cursor: wait; }
    .record { background: #ca2f3d; color: white; }
    .stop { background: #243447; color: white; }
    .meta { display: grid; gap: 8px; color: #3e4b58; }
    .notice { margin-top: 12px; min-height: 24px; font-weight: 700; color: #146c43; }
    ul { list-style: none; padding: 0; margin: 18px 0 0; display: grid; gap: 10px; }
    li { display: flex; align-items: center; justify-content: space-between; gap: 12px; border-top: 1px solid #e6ebf0; padding-top: 10px; }
    a { color: #0b5cad; font-weight: 700; overflow-wrap: anywhere; }
    @media (max-width: 540px) {
      header, li { align-items: flex-start; flex-direction: column; }
      .controls { grid-template-columns: 1fr; }
    }
    @media (prefers-color-scheme: dark) {
      body { background: #111820; color: #edf2f7; }
      .panel { background: #1b2530; border-color: #344354; box-shadow: none; }
      .status, .meta { color: #bac7d5; }
      li { border-top-color: #344354; }
      a { color: #8dc7ff; }
      .notice { color: #70d49b; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Cricket Audio Recorder</h1>
      <div class="status" id="status">Loading...</div>
    </header>
    <section class="panel">
      <div class="meta">
        <div>Current file: <strong id="file">-</strong></div>
        <div>Recorded data: <strong id="bytes">0</strong> bytes</div>
        <div>Device IP: <strong id="ip">-</strong></div>
        <div>Hardware: <strong id="hardware">Checking...</strong></div>
      </div>
      <div class="controls">
        <label for="label" style="display:flex; flex-direction:column; gap:6px; font-weight:700;">Label
          <select id="label" style="min-height:48px; border-radius:8px; border:1px solid #c7d2d8; padding:0 10px; font-size:1rem;">
            <option value="D1_Food_D2_Food">วันที่ 1 ให้อาหาร / วันที่ 2 ให้อาหาร</option>
            <option value="D1_NoFood_D2_Food">วันที่ 1 ไม่ให้อาหาร / วันที่ 2 ให้อาหาร</option>
            <option value="D1_Food_D2_NoFood">วันที่ 1 ให้อาหาร / วันที่ 2 ไม่ให้อาหาร</option>
            <option value="D1_NoFood_D2_NoFood">วันที่ 1 ไม่ให้อาหาร / วันที่ 2 ไม่ให้อาหาร</option>
          </select>
        </label>
        <button class="record" id="record">Record</button>
        <button class="stop" id="stop">Stop</button>
      </div>
      <div class="notice" id="notice"></div>
      <ul id="files"></ul>
    </section>
  </main>
  <script>
    const statusEl = document.querySelector('#status');
    const fileEl = document.querySelector('#file');
    const bytesEl = document.querySelector('#bytes');
    const ipEl = document.querySelector('#ip');
    const hardwareEl = document.querySelector('#hardware');
    const noticeEl = document.querySelector('#notice');
    const recordBtn = document.querySelector('#record');
    const stopBtn = document.querySelector('#stop');
    const labelEl = document.querySelector('#label');
    const filesEl = document.querySelector('#files');

    async function api(path) {
      const response = await fetch(path);
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function renderStatus(data) {
      statusEl.textContent = data.recording ? 'Recording' : 'Idle';
      fileEl.textContent = data.file || '-';
      bytesEl.textContent = data.bytes || 0;
      ipEl.textContent = data.ip || '-';
      hardwareEl.textContent = (data.sdReady ? 'SD ready' : 'SD error') + ' / ' + (data.i2sReady ? 'I2S ready' : 'I2S error');
      recordBtn.disabled = data.recording;
      stopBtn.disabled = !data.recording;
      noticeEl.textContent = data.recording ? 'Recording audio to SD card' : 'Ready';
      if (data.label) {
        labelEl.value = data.label;
      }
    }

    async function refreshFiles() {
      const files = await api('/files');
      filesEl.innerHTML = '';
      for (const file of files) {
        const li = document.createElement('li');
        const link = document.createElement('a');
        link.href = '/download?file=' + encodeURIComponent(file.name);
        link.textContent = file.name;
        const size = document.createElement('span');
        size.textContent = file.size + ' bytes';
        li.append(link, size);
        filesEl.append(li);
      }
    }

    async function refresh() {
      try {
        renderStatus(await api('/status'));
        await refreshFiles();
      } catch (error) {
        noticeEl.textContent = error.message;
      }
    }

    recordBtn.addEventListener('click', async () => {
      renderStatus(await api('/start?label=' + encodeURIComponent(labelEl.value)));
      await refreshFiles();
    });

    stopBtn.addEventListener('click', async () => {
      renderStatus(await api('/stop?label=' + encodeURIComponent(labelEl.value)));
      await refreshFiles();
    });

    refresh();
    setInterval(refresh, 1500);
  </script>
</body>
</html>
)HTML";

    server.send_P(200, "text/html", page);
  }

  void setupWebServer()
  {
    server.on("/", HTTP_GET, handleIndex);
    server.on("/status", HTTP_GET, sendJsonStatus);
    server.on("/start", HTTP_GET, handleStart);
    server.on("/stop", HTTP_GET, handleStop);
    server.on("/sync/start", HTTP_GET, handleRemoteStart);
    server.on("/sync/stop", HTTP_GET, handleRemoteStop);
    server.on("/files", HTTP_GET, handleFiles);
    server.on("/download", HTTP_GET, handleDownload);
    server.begin();
  }

  void recordAudioChunk()
  {
    if (!isRecording)
    {
      return;
    }

    size_t bytesRead = 0;
    esp_err_t result = i2s_read(I2S_NUM_0, i2sRaw, sizeof(i2sRaw), &bytesRead, 0);
    if (result != ESP_OK || bytesRead == 0)
    {
      return;
    }

    const size_t samplesRead = bytesRead / sizeof(i2sRaw[0]);
    for (size_t i = 0; i < samplesRead; i++)
    {
      int32_t sample = (i2sRaw[i] >> 16) * MIC_GAIN;
      sample = constrain(sample, -32768, 32767);
      pcmBuffer[i] = static_cast<int16_t>(sample);
    }

    const size_t bytesToWrite = samplesRead * sizeof(pcmBuffer[0]);
    if (recordingFile.write(reinterpret_cast<uint8_t *>(pcmBuffer), bytesToWrite) == bytesToWrite)
    {
      wavDataBytes += bytesToWrite;
    }
    else
    {
      Serial.println("SD write failed, stopping recording");
      stopRecording();
    }
  }
} // namespace

void setupWiFi()
{
  WiFi.disconnect(true, true);

  if (IS_MAIN_NODE)
  {
    WiFi.mode(WIFI_AP);
    bool apStarted = WiFi.softAP(MAIN_AP_SSID, MAIN_AP_PASSWORD);
    Serial.print("WiFi AP ");
    Serial.println(apStarted ? "started" : "failed");
    Serial.print("SSID: ");
    Serial.println(MAIN_AP_SSID);
    Serial.print("Password: ");
    Serial.println(MAIN_AP_PASSWORD);
    Serial.print("Open http://");
    Serial.println(WiFi.softAPIP());
  }
  else
  {
    WiFi.mode(WIFI_STA);
    Serial.printf("Connecting to %s...\n", MAIN_AP_SSID);
    WiFi.begin(MAIN_AP_SSID, MAIN_AP_PASSWORD);

    for (int attempt = 0; attempt < 30 && WiFi.status() != WL_CONNECTED; ++attempt)
    {
      delay(500);
      Serial.print('.');
    }

    Serial.println();
    if (WiFi.status() == WL_CONNECTED)
    {
      Serial.print("Connected to main node, IP: ");
      Serial.println(WiFi.localIP());
    }
    else
    {
      Serial.println("Failed to connect to main node");
    }
  }
}

void setup()
{
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("Starting Cricket Audio Recorder...");

  sdReady = SD.begin(SD_CS_PIN);
  if (!sdReady)
  {
    Serial.printf("SD card init failed on CS %d\n", SD_CS_PIN);
  }
  else
  {
    Serial.println("SD card initialized");
  }

  i2sReady = setupI2S();
  setupWiFi();
  setupWebServer();
}

void loop()
{
  server.handleClient();
  recordAudioChunk();
}
