#include <Arduino.h>
#include <HTTPClient.h>
#include <SD.h>
#include <SPI.h>
#include <WebServer.h>
#include <WiFi.h>
#include "driver/i2s.h"
#include <DHT.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

namespace
{
  // ==========================================
  // Config — เปลี่ยนก่อน flash แต่ละ node
  // ==========================================
  constexpr bool IS_MAIN_NODE = true; // false สำหรับ secondary
  constexpr int  NODE_ID      = 0;    // 0 = main, 1, 2 = secondary

  // --- Pins ---
  constexpr int MIC_WS_PIN  = 42;
  constexpr int MIC_SCK_PIN = 41;
  constexpr int MIC_SD_PIN  = 40;
  constexpr int SD_CS_PIN   = 10;
  constexpr int DHT_PIN     = 4;
  DHT dht(DHT_PIN, DHT22);

  // --- Audio ---
  constexpr uint32_t SAMPLE_RATE   = 16000;
  constexpr size_t   I2S_SAMPLES   = 512;
  constexpr int16_t  MIC_GAIN      = 4;
  constexpr uint32_t MAX_FILE_SIZE = 100UL * 1024 * 1024;

  // --- WiFi ---
  constexpr char AP_SSID[]      = "Cricket-Audio";
  constexpr char AP_PASSWORD[]  = "12345678";
  constexpr char MAIN_NODE_IP[] = "192.168.4.1";

  // --- Multi-node (main only) ---
  constexpr int MAX_SECONDARY = 2;
  String secondaryIPs[MAX_SECONDARY]; // index = NODE_ID - 1
  int    secondaryCount = 0;

  // --- State ---
  WebServer server(80);
  File      recordingFile;
  String    currentFolder;
  String    recordingPath;
  String    dhtLogPath;
  String    recordingLabel  = "default";
  volatile bool isRecording = false;
  bool      sdReady         = false;
  bool      i2sReady        = false;
  uint32_t  wavDataBytes    = 0;
  int       audioFileIndex  = 0;
  SemaphoreHandle_t sdMutex;

  unsigned long lastDhtReadTime = 0;
  constexpr unsigned long DHT_INTERVAL = 10000;

  int32_t i2sRaw[I2S_SAMPLES];
  int16_t pcmBuffer[I2S_SAMPLES];

  // --- WAV Helpers ---
  void writeLE16(File &f, uint16_t v)
  {
    f.write(v & 0xff);
    f.write((v >> 8) & 0xff);
  }
  void writeLE32(File &f, uint32_t v)
  {
    f.write(v & 0xff);
    f.write((v >> 8) & 0xff);
    f.write((v >> 16) & 0xff);
    f.write((v >> 24) & 0xff);
  }
  void writeWavHeader(File &f, uint32_t dataBytes)
  {
    f.seek(0);
    f.write((const uint8_t *)"RIFF", 4); writeLE32(f, 36 + dataBytes);
    f.write((const uint8_t *)"WAVE", 4);
    f.write((const uint8_t *)"fmt ", 4); writeLE32(f, 16);
    writeLE16(f, 1); writeLE16(f, 1);
    writeLE32(f, SAMPLE_RATE); writeLE32(f, SAMPLE_RATE * 2);
    writeLE16(f, 2); writeLE16(f, 16);
    f.write((const uint8_t *)"data", 4); writeLE32(f, dataBytes);
  }

  String sanitizeLabel(const String &label)
  {
    String s = label;
    s.replace(' ', '_'); s.replace('/', '_'); s.replace('\\', '_');
    return s.length() == 0 ? "default" : s;
  }

  String makeAudioPath() { return currentFolder + "/audio_" + String(audioFileIndex++) + ".wav"; }
  String makeDhtPath()   { return currentFolder + "/data_"  + String(audioFileIndex - 1) + ".csv"; }

  bool openNextAudioFile()
  {
    recordingPath = makeAudioPath();
    dhtLogPath    = makeDhtPath();
    File dhtF = SD.open(dhtLogPath, FILE_WRITE);
    if (dhtF) { dhtF.println("Timestamp_ms,Temperature_C,Humidity_pct"); dhtF.close(); }
    recordingFile = SD.open(recordingPath, FILE_WRITE);
    if (!recordingFile) return false;
    wavDataBytes = 0;
    writeWavHeader(recordingFile, 0);
    return true;
  }

  // --- Recording Control ---
  bool startRecording(const String &label = "default")
  {
    if (isRecording) return true;
    if (!sdReady || !i2sReady) return false;
    if (xSemaphoreTake(sdMutex, portMAX_DELAY) != pdTRUE) return false;

    recordingLabel = sanitizeLabel(label);
    audioFileIndex = 0;
    currentFolder  = "/node" + String(NODE_ID) + "_" + recordingLabel + "_" + String(millis());
    SD.mkdir(currentFolder);

    File meta = SD.open(currentFolder + "/meta.txt", FILE_WRITE);
    if (meta)
    {
      meta.println("Node="  + String(NODE_ID));
      meta.println("Label=" + recordingLabel);
      meta.println("Start_ms=" + String(millis()));
      meta.close();
    }

    if (!openNextAudioFile()) { xSemaphoreGive(sdMutex); return false; }

    isRecording = true;
    xSemaphoreGive(sdMutex);
    Serial.println("[Node" + String(NODE_ID) + "] Recording: " + recordingPath);
    return true;
  }

  void stopRecording()
  {
    if (!isRecording) return;
    if (xSemaphoreTake(sdMutex, portMAX_DELAY) != pdTRUE) return;
    isRecording = false;
    writeWavHeader(recordingFile, wavDataBytes);
    recordingFile.close();
    File meta = SD.open(currentFolder + "/meta.txt", FILE_APPEND);
    if (meta) { meta.println("End_ms=" + String(millis())); meta.close(); }
    xSemaphoreGive(sdMutex);
    Serial.println("[Node" + String(NODE_ID) + "] Stopped");
  }

  // --- Audio Task (Core 0) ---
  void recordAudioChunk()
  {
    if (!isRecording) return;
    size_t bytesRead = 0;
    if (i2s_read(I2S_NUM_0, i2sRaw, sizeof(i2sRaw), &bytesRead, portMAX_DELAY) != ESP_OK || bytesRead == 0) return;

    const size_t samplesRead = bytesRead / sizeof(i2sRaw[0]);
    for (size_t i = 0; i < samplesRead; i++)
    {
      int32_t s = (i2sRaw[i] >> 16) * MIC_GAIN;
      pcmBuffer[i] = static_cast<int16_t>(constrain(s, -32768, 32767));
    }
    const size_t bytesToWrite = samplesRead * sizeof(pcmBuffer[0]);

    if (xSemaphoreTake(sdMutex, portMAX_DELAY) != pdTRUE) return;

    if (recordingFile.write(reinterpret_cast<uint8_t *>(pcmBuffer), bytesToWrite) != bytesToWrite)
    {
      Serial.println("SD write failed, stopping");
      xSemaphoreGive(sdMutex);
      stopRecording();
      return;
    }

    wavDataBytes += bytesToWrite;
    if (wavDataBytes >= MAX_FILE_SIZE)
    {
      writeWavHeader(recordingFile, wavDataBytes);
      recordingFile.close();
      Serial.println("Split: opening next file");
      if (!openNextAudioFile()) isRecording = false;
    }

    xSemaphoreGive(sdMutex);
  }

  void audioTask(void *) { for (;;) { recordAudioChunk(); vTaskDelay(1); } }

  // --- Multi-node Broadcast (Main only) ---
  void broadcastToSecondaries(const String &path)
  {
    for (int i = 0; i < secondaryCount; i++)
    {
      HTTPClient http;
      String url = "http://" + secondaryIPs[i] + path;
      http.begin(url);
      http.setConnectTimeout(2000);
      http.setTimeout(2000);
      int code = http.GET();
      Serial.println("Sync -> " + url + " [" + String(code) + "]");
      http.end();
    }
  }

  // --- Web Handlers ---
  void sendJsonStatus()
  {
    String ip = IS_MAIN_NODE ? WiFi.softAPIP().toString() : WiFi.localIP().toString();
    String json = "{";
    json += "\"recording\":"  + String(isRecording ? "true" : "false") + ",";
    json += "\"file\":\""     + recordingPath + "\",";
    json += "\"bytes\":"      + String(wavDataBytes) + ",";
    json += "\"ip\":\""       + ip + "\",";
    json += "\"sdReady\":"    + String(sdReady  ? "true" : "false") + ",";
    json += "\"i2sReady\":"   + String(i2sReady ? "true" : "false") + ",";
    json += "\"label\":\""    + recordingLabel + "\",";
    json += "\"node\":"       + String(NODE_ID) + ",";
    json += "\"nodes\":"      + String(secondaryCount + 1) + ",";
    json += "\"secondaries\":[";
    for (int i = 0; i < secondaryCount; i++)
    {
      if (i > 0) json += ",";
      json += "\"" + secondaryIPs[i] + "\"";
    }
    json += "]}";
    server.send(200, "application/json", json);
  }

  // Main node: start own recording then broadcast to secondaries
  void handleStart()
  {
    String label = server.hasArg("label") ? server.arg("label") : "default";
    startRecording(label);
    if (IS_MAIN_NODE)
      broadcastToSecondaries("/sync/start?label=" + label);
    sendJsonStatus();
  }

  // Main node: stop own recording then broadcast to secondaries
  void handleStop()
  {
    stopRecording();
    if (IS_MAIN_NODE)
      broadcastToSecondaries("/sync/stop");
    sendJsonStatus();
  }

  // Secondary nodes: called by main via broadcast
  void handleSyncStart()
  {
    String label = server.hasArg("label") ? server.arg("label") : "default";
    startRecording(label);
    sendJsonStatus();
  }
  void handleSyncStop()
  {
    stopRecording();
    sendJsonStatus();
  }

  // Main node: secondary calls this on boot (and on reconnect) to register its IP
  void handleRegister()
  {
    if (!server.hasArg("ip") || !server.hasArg("node"))
    {
      server.send(400, "text/plain", "Missing ip or node"); return;
    }
    String ip  = server.arg("ip");
    int    nid = server.arg("node").toInt();
    if (nid < 1 || nid > MAX_SECONDARY) { server.send(400, "text/plain", "Invalid node"); return; }

    int slot = nid - 1;
    if (secondaryIPs[slot] != ip)
    {
      secondaryIPs[slot] = ip;
      if (slot >= secondaryCount) secondaryCount = slot + 1;
      Serial.println("Registered node" + String(nid) + " -> " + ip);
    }
    server.send(200, "text/plain", "OK");
  }

  void handleFiles()
  {
    String json = "[";
    bool first = true;
    if (xSemaphoreTake(sdMutex, portMAX_DELAY) == pdTRUE)
    {
      File root   = SD.open("/");
      File folder = root.openNextFile();
      while (folder)
      {
        if (folder.isDirectory())
        {
          File file = folder.openNextFile();
          while (file)
          {
            if (!first) json += ",";
            json += "{\"name\":\"/" + String(folder.name()) + "/" + String(file.name()) + "\",\"size\":" + String(file.size()) + "}";
            first = false;
            file  = folder.openNextFile();
          }
        }
        folder = root.openNextFile();
      }
      xSemaphoreGive(sdMutex);
    }
    server.send(200, "application/json", json + "]");
  }

  void handleDownload()
  {
    if (!server.hasArg("file")) { server.send(400, "text/plain", "Missing file param"); return; }
    String path = server.arg("file");
    if (!path.startsWith("/")) path = "/" + path;
    if (xSemaphoreTake(sdMutex, portMAX_DELAY) == pdTRUE)
    {
      File f = SD.open(path, FILE_READ);
      if (!f) { xSemaphoreGive(sdMutex); server.send(404, "text/plain", "Not found"); return; }
      server.streamFile(f, "application/octet-stream");
      f.close();
      xSemaphoreGive(sdMutex);
    }
  }

  void handleIndex()
  {
    static const char page[] PROGMEM = R"HTML(
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cricket Audio Logger</title>
  <style>
    :root { color-scheme: light dark; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; min-height: 100vh; background: #f3f5f7; color: #18202a; }
    main { width: min(760px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 20px; }
    h1 { font-size: clamp(1.55rem, 5vw, 2.25rem); margin: 0; }
    .status { font-weight: 700; color: #516171; }
    .panel { background: #fff; border: 1px solid #d8e0e8; border-radius: 8px; padding: 18px; box-shadow: 0 8px 26px rgba(24,32,42,0.08); margin-bottom: 16px; }
    .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 16px 0; }
    button { min-height: 48px; border: 0; border-radius: 8px; font-size: 1rem; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: 0.45; cursor: wait; }
    .record { background: #ca2f3d; color: white; }
    .stop   { background: #243447; color: white; }
    .meta   { display: grid; gap: 8px; color: #3e4b58; }
    .notice { margin-top: 12px; min-height: 24px; font-weight: 700; color: #146c43; }
    .nodes  { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }
    .node-badge { padding: 2px 10px; border-radius: 99px; font-size: 0.82rem; font-weight: 700; background: #e3f0ff; color: #0b5cad; }
    .node-badge.self { background: #d4edda; color: #146c43; }
    ul { list-style: none; padding: 0; margin: 18px 0 0; display: grid; gap: 10px; }
    li { display: flex; align-items: center; justify-content: space-between; gap: 12px; border-top: 1px solid #e6ebf0; padding-top: 10px; }
    a  { color: #0b5cad; font-weight: 700; overflow-wrap: anywhere; }
    @media (max-width: 540px) {
      header, li { align-items: flex-start; flex-direction: column; }
      .controls  { grid-template-columns: 1fr; }
    }
    @media (prefers-color-scheme: dark) {
      body   { background: #111820; color: #edf2f7; }
      .panel { background: #1b2530; border-color: #344354; box-shadow: none; }
      .status, .meta { color: #bac7d5; }
      li { border-top-color: #344354; }
      a  { color: #8dc7ff; }
      .notice { color: #70d49b; }
      .node-badge      { background: #1a3356; color: #8dc7ff; }
      .node-badge.self { background: #0e3320; color: #70d49b; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <h1>Cricket Audio Logger</h1>
    <div class="status" id="status">Loading...</div>
  </header>

  <section class="panel">
    <div class="meta">
      <div>Nodes online: <span class="nodes" id="nodeList"></span></div>
      <div>Current file: <strong id="file">-</strong></div>
      <div>Recorded data: <strong id="bytes">0</strong> bytes <i>(Auto-split at 100 MB)</i></div>
      <div>Device IP: <strong id="ip">-</strong></div>
      <div>Hardware: <strong id="hardware">Checking...</strong></div>
    </div>
    <div class="controls">
      <label for="label" style="display:flex;flex-direction:column;gap:6px;font-weight:700;">Label
        <select id="label" style="min-height:48px;border-radius:8px;border:1px solid #c7d2d8;padding:0 10px;font-size:1rem;">
          <option value="D1_Food_D2_Food">วันที่ 1 ให้อาหาร / วันที่ 2 ให้อาหาร</option>
          <option value="D1_NoFood_D2_Food">วันที่ 1 ไม่ให้อาหาร / วันที่ 2 ให้อาหาร</option>
          <option value="D1_Food_D2_NoFood">วันที่ 1 ให้อาหาร / วันที่ 2 ไม่ให้อาหาร</option>
          <option value="D1_NoFood_D2_NoFood">วันที่ 1 ไม่ให้อาหาร / วันที่ 2 ไม่ให้อาหาร</option>
        </select>
      </label>
      <button class="record" id="record">Record All</button>
      <button class="stop"   id="stop">Stop All</button>
    </div>
    <div class="notice" id="notice"></div>
  </section>

  <section class="panel">
    <strong>Files on this node</strong>
    <ul id="files"></ul>
  </section>
</main>
<script>
  const statusEl  = document.querySelector('#status');
  const fileEl    = document.querySelector('#file');
  const bytesEl   = document.querySelector('#bytes');
  const ipEl      = document.querySelector('#ip');
  const hardwareEl= document.querySelector('#hardware');
  const noticeEl  = document.querySelector('#notice');
  const recordBtn = document.querySelector('#record');
  const stopBtn   = document.querySelector('#stop');
  const labelEl   = document.querySelector('#label');
  const filesEl   = document.querySelector('#files');
  const nodeListEl= document.querySelector('#nodeList');

  async function api(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }

  function renderStatus(d) {
    statusEl.textContent  = d.recording ? 'Recording' : 'Idle';
    fileEl.textContent    = d.file  || '-';
    bytesEl.textContent   = d.bytes || 0;
    ipEl.textContent      = d.ip    || '-';
    hardwareEl.textContent= (d.sdReady ? 'SD OK' : 'SD ERR') + ' / ' + (d.i2sReady ? 'I2S OK' : 'I2S ERR');
    recordBtn.disabled    = d.recording;
    stopBtn.disabled      = !d.recording;
    noticeEl.textContent  = d.recording
      ? `Recording on ${d.nodes} node(s)...`
      : `Ready — ${d.nodes - 1} secondary node(s) connected`;
    if (d.label) labelEl.value = d.label;

    nodeListEl.innerHTML = '';
    const self = document.createElement('span');
    self.className = 'node-badge self';
    self.textContent = 'Main (' + d.ip + ')';
    nodeListEl.appendChild(self);
    (d.secondaries || []).forEach(ip => {
      const b = document.createElement('span');
      b.className = 'node-badge';
      b.textContent = ip;
      nodeListEl.appendChild(b);
    });
  }

  async function refreshFiles() {
    const files = await api('/files');
    filesEl.innerHTML = '';
    for (const f of files) {
      const li   = document.createElement('li');
      const link = document.createElement('a');
      link.href        = '/download?file=' + encodeURIComponent(f.name);
      link.textContent = f.name;
      const size = document.createElement('span');
      size.textContent = f.size + ' bytes';
      li.append(link, size);
      filesEl.append(li);
    }
  }

  async function refresh() {
    try { renderStatus(await api('/status')); await refreshFiles(); }
    catch (e) { noticeEl.textContent = e.message; }
  }

  recordBtn.addEventListener('click', async () => {
    renderStatus(await api('/start?label=' + encodeURIComponent(labelEl.value)));
    await refreshFiles();
  });
  stopBtn.addEventListener('click', async () => {
    renderStatus(await api('/stop'));
    await refreshFiles();
  });

  refresh();
  setInterval(refresh, 2500);
</script>
</body>
</html>
)HTML";
    server.send_P(200, "text/html", page);
  }

  void setupWebServer()
  {
    // ทุก node มี endpoints เหล่านี้
    server.on("/status",     HTTP_GET, sendJsonStatus);
    server.on("/sync/start", HTTP_GET, handleSyncStart);
    server.on("/sync/stop",  HTTP_GET, handleSyncStop);
    server.on("/files",      HTTP_GET, handleFiles);
    server.on("/download",   HTTP_GET, handleDownload);

    // Main node เพิ่ม UI และ registration endpoint
    if (IS_MAIN_NODE)
    {
      server.on("/",         HTTP_GET, handleIndex);
      server.on("/start",    HTTP_GET, handleStart);
      server.on("/stop",     HTTP_GET, handleStop);
      server.on("/register", HTTP_GET, handleRegister);
    }

    server.begin();
  }

} // namespace

// ==========================================
// I2S Setup
// ==========================================
bool setupI2S()
{
  i2s_config_t cfg = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate          = SAMPLE_RATE,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count        = 4,
    .dma_buf_len          = (int)I2S_SAMPLES,
    .use_apll             = false,
    .tx_desc_auto_clear   = false,
    .fixed_mclk           = 0
  };
  i2s_pin_config_t pins = {
    .bck_io_num   = MIC_SCK_PIN,
    .ws_io_num    = MIC_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = MIC_SD_PIN
  };
  if (i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL) != ESP_OK) return false;
  if (i2s_set_pin(I2S_NUM_0, &pins) != ESP_OK) return false;
  i2s_zero_dma_buffer(I2S_NUM_0);
  return true;
}

// ==========================================
// WiFi Setup
// ==========================================
void setupWiFi()
{
  WiFi.disconnect(true, true);

  if (IS_MAIN_NODE)
  {
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, AP_PASSWORD);
    Serial.print("AP started — http://");
    Serial.println(WiFi.softAPIP());
  }
  else
  {
    WiFi.mode(WIFI_STA);
    WiFi.begin(AP_SSID, AP_PASSWORD);
    Serial.print("Connecting to main node");
    for (int i = 0; i < 30 && WiFi.status() != WL_CONNECTED; i++)
    {
      delay(500);
      Serial.print('.');
    }
    Serial.println();
    if (WiFi.status() == WL_CONNECTED)
      Serial.println("Connected, IP: " + WiFi.localIP().toString());
    else
      Serial.println("WiFi connection failed");
  }
}

// Secondary node: sync state กับ Main หลัง register สำเร็จ
void syncStateWithMain()
{
  HTTPClient http;
  http.begin("http://" + String(MAIN_NODE_IP) + "/status");
  http.setConnectTimeout(2000);
  int code = http.GET();
  if (code != 200) { http.end(); return; }
  String body = http.getString();
  http.end();

  bool mainRecording = body.indexOf("\"recording\":true") >= 0;

  if (isRecording && !mainRecording)
  {
    Serial.println("[Node" + String(NODE_ID) + "] Main stopped — stopping local recording");
    stopRecording();
  }
  else if (!isRecording && mainRecording)
  {
    int li = body.indexOf("\"label\":\"");
    if (li >= 0)
    {
      int start = li + 9;
      int end   = body.indexOf("\"", start);
      String label = body.substring(start, end);
      Serial.println("[Node" + String(NODE_ID) + "] Main is recording — joining: " + label);
      startRecording(label);
    }
  }
}

// Secondary node: แจ้ง IP และ NODE_ID ให้ main node ทราบ (retry 5 ครั้ง)
void registerWithMain()
{
  if (IS_MAIN_NODE || WiFi.status() != WL_CONNECTED) return;
  String url = "http://" + String(MAIN_NODE_IP) +
               "/register?ip=" + WiFi.localIP().toString() +
               "&node=" + String(NODE_ID);
  for (int attempt = 0; attempt < 5; attempt++)
  {
    HTTPClient http;
    http.begin(url);
    http.setConnectTimeout(2000);
    int code = http.GET();
    http.end();
    if (code == 200)
    {
      Serial.println("Registered with main node (attempt " + String(attempt + 1) + ")");
      syncStateWithMain();
      return;
    }
    delay(1000);
  }
  Serial.println("Failed to register with main node");
}

// ==========================================
// Setup & Loop
// ==========================================
void setup()
{
  Serial.begin(115200);
  sdMutex  = xSemaphoreCreateMutex();
  dht.begin();
  sdReady  = SD.begin(SD_CS_PIN);
  i2sReady = setupI2S();
  Serial.printf("[Node%d] SD:%s I2S:%s\n", NODE_ID, sdReady ? "OK" : "FAIL", i2sReady ? "OK" : "FAIL");

  if (!IS_MAIN_NODE)
  {
    WiFi.onEvent([](WiFiEvent_t, WiFiEventInfo_t) { registerWithMain(); },
                 ARDUINO_EVENT_WIFI_STA_GOT_IP);
  }

  setupWiFi();
  registerWithMain();   // no-op on main node
  setupWebServer();
  xTaskCreatePinnedToCore(audioTask, "Audio", 8192, NULL, 2, NULL, 0);
}

void loop()
{
  server.handleClient();

  if (millis() - lastDhtReadTime >= DHT_INTERVAL)
  {
    lastDhtReadTime = millis();
    float h = dht.readHumidity();
    float t = dht.readTemperature();

    if (isnan(h) || isnan(t)) { Serial.println("DHT read failed"); return; }
    Serial.printf("[Node%d] DHT: %.1f C  %.1f %%\n", NODE_ID, t, h);

    if (isRecording && xSemaphoreTake(sdMutex, portMAX_DELAY) == pdTRUE)
    {
      File f = SD.open(dhtLogPath, FILE_APPEND);
      if (f) { f.printf("%lu,%.2f,%.2f\n", millis(), t, h); f.close(); }
      xSemaphoreGive(sdMutex);
    }
  }
}
