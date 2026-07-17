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
#include <Preferences.h>

namespace
{
  // ==========================================
  // Config — เปลี่ยนก่อน flash แต่ละ node
  // ==========================================
  constexpr bool IS_MAIN_NODE = false; // false สำหรับ secondary
  constexpr int NODE_ID = 2;           // 0 = main, 1, 2 = secondary

  // Growth stage ของบ่อที่ node นี้ติดตั้งอยู่ (1 บ่อ = 1 stage) — เลือก/แก้ได้ผ่านหน้าเว็บของ node นั้นๆ
  // โดยไม่ต้อง reflash (เก็บใน NVS ผ่าน Preferences) ค่านี้คือ default ตอน flash ครั้งแรกเท่านั้น
  // ใช้เลข 1-4 แทนชื่อ stage ไปก่อน เพราะ taxonomy ยัง unconfirmed จากผู้เชี่ยวชาญ
  constexpr char DEFAULT_STAGE_LABEL[] = "1";
  constexpr const char *STAGE_OPTIONS[] = {"1", "2", "3", "4"};
  constexpr size_t STAGE_OPTIONS_COUNT = 4;

  // --- Pins ---
  constexpr int MIC_WS_PIN = 42;
  constexpr int MIC_SCK_PIN = 41;
  constexpr int MIC_SD_PIN = 40;
  constexpr int SD_CS_PIN = 10;
  constexpr int DHT_PIN = 14;
  constexpr bool DHT_ENABLED = true;
  constexpr bool DHT_SD_LOGGING = true; // ปิดชั่วคราวเพื่อแยกปัญหาว่าการเปิดไฟล์ CSV แทรกกลางไฟล์เสียงที่กำลังเขียน ทำให้ write fail หรือเปล่า
  DHT dht(DHT_PIN, DHT11);

  // --- Audio ---
  constexpr uint32_t SAMPLE_RATE = 22050; // ตรงกับ Python pipeline (sr=22050)
  constexpr size_t I2S_SAMPLES = 1024;
  constexpr uint32_t MAX_FILE_SIZE = 100UL * 1024 * 1024;

  // --- AGC ---
  constexpr int16_t AGC_GAIN_MIN = 1;
  constexpr int16_t AGC_GAIN_MAX = 16;
  constexpr float AGC_TARGET_RMS = 2500.0f; // target ≈ -22 dBFS (ไม่ clip)
  constexpr float AGC_ALPHA = 0.05f;        // EMA smoothing (ปรับช้า → ป้องกัน oscillation)
  int16_t micGain = 4;                      // ปรับได้ runtime (ไม่ใช่ constexpr)

  // --- Adaptive noise floor ---
  // แทนที่จะ calibrate ครั้งเดียวตอน start (ต้องเงียบ ซึ่งบังคับจิ้งหรีดไม่ได้) ให้ไล่ตาม noise floor
  // ต่อเนื่องด้วย asymmetric EMA ตลอดเวลาที่ isRecording: ตามลงไวเมื่อเจอช่วงเงียบจริงระหว่าง chirp,
  // ตามขึ้นช้ามากกันไม่ให้ chirp เองดันพื้นเสียงขึ้นเรื่อยๆ
  constexpr float NOISE_FLOOR_ALPHA_DOWN = 0.05f;
  constexpr float NOISE_FLOOR_ALPHA_UP = 0.0005f;

  // --- Auto-record (trigger on sound, stop after prolonged silence) ---
  // ยังไม่มีข้อมูลภาคสนามเรื่องช่วงเงียบระหว่าง calling bout ของจิ้งหรีดไทยจริงๆ
  // ค่าเริ่มต้นอิงจากงานวิจัย Gryllus (calling bout ยาวได้ถึง ~2 นาที, เสียง chirp ภายใน bout ห่างกัน <0.5s)
  // ตั้ง SILENCE_HANG_MS ให้ยาวพอไม่ตัดกลาง bout แต่ปรับได้ตามข้อมูลจริงหน้างานทีหลัง
  constexpr float THRESHOLD_MULTIPLIER = 3.0f;     // เสียงต้องดังกว่า noise floor กี่เท่าถึงเริ่มอัด (trigger)
  constexpr unsigned long SILENCE_HANG_MS = 15000; // เงียบติดต่อกันนานเท่านี้ถึงหยุดไฟล์ (default 15s)
  constexpr size_t PRE_BUFFER_CHUNKS = 15;         // ~0.7s ก่อน trigger เก็บไว้กันเสียงต้น chirp หาย

  // --- WiFi ---
  constexpr char AP_SSID[] = "Cricket-Audio";
  constexpr char AP_PASSWORD[] = "12345678";
  constexpr char MAIN_NODE_IP[] = "192.168.4.1";

  // --- Multi-node (main only) ---
  constexpr int MAX_SECONDARY = 2;
  String secondaryIPs[MAX_SECONDARY];   // index = NODE_ID - 1
  String secondaryStage[MAX_SECONDARY]; // stage ล่าสุดที่ secondary push มา
  float secondaryTemp[MAX_SECONDARY] = {0};
  float secondaryHumidity[MAX_SECONDARY] = {0};
  bool secondaryRecording[MAX_SECONDARY] = {false};
  bool secondarySdReady[MAX_SECONDARY] = {false};
  int secondaryCount = 0;

  // --- State ---
  WebServer server(80);
  File recordingFile;
  String currentFolder;
  String recordingPath;
  String dhtLogPath;
  String recordingLabel = "default";
  Preferences stagePrefs;
  String currentStageLabel;          // stage ปัจจุบันของ node นี้ โหลดจาก NVS ตอน boot, แก้ผ่าน /stage/set
  volatile bool isRecording = false; // session ทั้งหมด (ตั้งแต่กด start ถึง stop) ไม่ใช่แค่ตอนกำลังเขียนไฟล์
  volatile bool burstActive = false; // true เฉพาะตอนกำลังเขียนไฟล์เสียง (มีเสียงจริงเกิน threshold)
  bool sdReady = false;
  bool i2sReady = false;
  uint32_t wavDataBytes = 0;
  int audioFileIndex = 0;
  SemaphoreHandle_t sdMutex;

  float noiseFloorRms = 0.0f;         // เสียงพื้นหลัง ปรับต่อเนื่องด้วย adaptive EMA ระหว่างอัด ไม่ใช่ calibrate ครั้งเดียวตอน start
  bool noiseFloorInitialized = false; // true หลังมี chunk แรกเข้ามา seed ค่าเริ่มต้น (persist ข้าม start/stop session)
  float soundThreshold = 0.0f;        // = noiseFloorRms * THRESHOLD_MULTIPLIER, คำนวณใหม่ทุก chunk
  unsigned long lastLoudMs = 0;       // เวลาล่าสุดที่ RMS เกิน threshold (ใช้กับ silence-hang timer)

  unsigned long lastDhtReadTime = 0;
  constexpr unsigned long DHT_INTERVAL = 10000;
  float lastTemperature = 0.0f;
  float lastHumidity = 0.0f;
  float lastRms = 0.0f; // ค่า RMS ล่าสุดจาก audioTask ใช้โชว์กราฟระดับเสียง

  // --- History สำหรับกราฟบนเว็บ ---
  // Temp/Humidity เปลี่ยนช้า เก็บทุก 10s ย้อนหลัง ~15 นาที
  constexpr size_t ENV_HISTORY_LEN = 90;
  constexpr unsigned long ENV_HISTORY_INTERVAL = 10000;
  unsigned long lastEnvHistoryPushMs = 0;
  float tempHistory[ENV_HISTORY_LEN] = {0};
  float humidityHistory[ENV_HISTORY_LEN] = {0};
  size_t envHistoryIdx = 0;
  size_t envHistoryFilled = 0;

  // เสียงเปลี่ยนเร็วกว่ามาก เก็บทุก 1s ย้อนหลัง ~5 นาที ให้กราฟดู real-time กว่า
  constexpr size_t RMS_HISTORY_LEN = 300;
  constexpr unsigned long RMS_HISTORY_INTERVAL = 1000;
  unsigned long lastRmsHistoryPushMs = 0;
  float rmsHistory[RMS_HISTORY_LEN] = {0};
  size_t rmsHistoryIdx = 0;
  size_t rmsHistoryFilled = 0;

  void pushEnvHistory()
  {
    tempHistory[envHistoryIdx] = lastTemperature;
    humidityHistory[envHistoryIdx] = lastHumidity;
    envHistoryIdx = (envHistoryIdx + 1) % ENV_HISTORY_LEN;
    if (envHistoryFilled < ENV_HISTORY_LEN)
      envHistoryFilled++;
  }

  void pushRmsHistory()
  {
    rmsHistory[rmsHistoryIdx] = lastRms;
    rmsHistoryIdx = (rmsHistoryIdx + 1) % RMS_HISTORY_LEN;
    if (rmsHistoryFilled < RMS_HISTORY_LEN)
      rmsHistoryFilled++;
  }

  int32_t i2sRaw[I2S_SAMPLES];
  int16_t pcmBuffer[I2S_SAMPLES];

  // --- Pre-trigger ring buffer (เก็บเสียงล่าสุดไว้ตอน "ฟังเฉยๆ" เผื่อ trigger จะได้ไม่เสียเสียงต้น chirp) ---
  int16_t preBuffer[PRE_BUFFER_CHUNKS][I2S_SAMPLES];
  size_t preBufSampleCount[PRE_BUFFER_CHUNKS] = {0};
  size_t preBufIdx = 0;
  size_t preBufFilled = 0;

  void resetPreBuffer()
  {
    preBufIdx = 0;
    preBufFilled = 0;
  }

  void pushPreBuffer(const int16_t *buf, size_t count)
  {
    memcpy(preBuffer[preBufIdx], buf, count * sizeof(int16_t));
    preBufSampleCount[preBufIdx] = count;
    preBufIdx = (preBufIdx + 1) % PRE_BUFFER_CHUNKS;
    if (preBufFilled < PRE_BUFFER_CHUNKS)
      preBufFilled++;
  }

  // เขียน chunk ที่สะสมไว้ก่อน trigger ลงไฟล์ที่เพิ่งเปิด (เรียงเก่า -> ใหม่) — ต้องถือ sdMutex อยู่แล้วตอนเรียก
  void flushPreBufferToFile()
  {
    size_t start = (preBufFilled < PRE_BUFFER_CHUNKS) ? 0 : preBufIdx;
    for (size_t n = 0; n < preBufFilled; n++)
    {
      size_t idx = (start + n) % PRE_BUFFER_CHUNKS;
      size_t bytes = preBufSampleCount[idx] * sizeof(int16_t);
      recordingFile.write(reinterpret_cast<uint8_t *>(preBuffer[idx]), bytes);
      wavDataBytes += bytes;
    }
    resetPreBuffer();
  }

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
    f.write((const uint8_t *)"RIFF", 4);
    writeLE32(f, 36 + dataBytes);
    f.write((const uint8_t *)"WAVE", 4);
    f.write((const uint8_t *)"fmt ", 4);
    writeLE32(f, 16);
    writeLE16(f, 1);
    writeLE16(f, 1);
    writeLE32(f, SAMPLE_RATE);
    writeLE32(f, SAMPLE_RATE * 2);
    writeLE16(f, 2);
    writeLE16(f, 16);
    f.write((const uint8_t *)"data", 4);
    writeLE32(f, dataBytes);
  }

  String sanitizeLabel(const String &label)
  {
    String s = label;
    s.replace(' ', '_');
    s.replace('/', '_');
    s.replace('\\', '_');
    return s.length() == 0 ? "default" : s;
  }

  String makeAudioPath() { return currentFolder + "/audio_" + String(audioFileIndex++) + ".wav"; }
  String makeDhtPath() { return currentFolder + "/data_" + String(audioFileIndex - 1) + ".csv"; }

  bool openNextAudioFile()
  {
    recordingPath = makeAudioPath();
    dhtLogPath = makeDhtPath();
    File dhtF = SD.open(dhtLogPath, FILE_WRITE);
    if (dhtF)
    {
      dhtF.println("Timestamp_ms,Temperature_C,Humidity_pct");
      dhtF.close();
    }
    recordingFile = SD.open(recordingPath, FILE_WRITE);
    if (!recordingFile)
      return false;
    wavDataBytes = 0;
    writeWavHeader(recordingFile, 0);
    return true;
  }

  // --- Recording Control ---
  // Label มาจาก currentStageLabel ของ node นี้เสมอ (ตั้งผ่าน /stage/set) ไม่รับ label จาก request/broadcast
  // เพราะแต่ละ node อาจอยู่คนละบ่อ/คนละ stage กัน
  bool startRecording()
  {
    if (isRecording)
      return true;
    if (!sdReady || !i2sReady)
      return false;

    if (xSemaphoreTake(sdMutex, portMAX_DELAY) != pdTRUE)
      return false;

    recordingLabel = sanitizeLabel(currentStageLabel);
    audioFileIndex = 0;
    recordingPath = ""; // ยังไม่มีไฟล์จนกว่าจะมีเสียงเกิน threshold ครั้งแรก
    dhtLogPath = "";    // กัน DHT logger เขียนทับไฟล์ของ session ก่อนหน้าตอนยังไม่ trigger
    currentFolder = "/node" + String(NODE_ID) + "_" + recordingLabel + "_" + String(millis());
    SD.mkdir(currentFolder);

    File meta = SD.open(currentFolder + "/meta.txt", FILE_WRITE);
    if (meta)
    {
      meta.println("Node=" + String(NODE_ID));
      meta.println("Label=" + recordingLabel);
      meta.println("Start_ms=" + String(millis()));
      // ค่า adaptive ล่าสุด ณ ตอน start (0 ถ้ายังไม่เคยมีเสียงเข้าไมค์เลยตั้งแต่ boot) — ค่านี้จะขยับต่อระหว่างอัด ดูค่าสุดท้ายได้ตอน End
      meta.println("NoiseFloorRMS_atStart=" + String(noiseFloorRms, 2));
      meta.println("ThresholdMultiplier=" + String(THRESHOLD_MULTIPLIER, 2));
      meta.println("SilenceHangMs=" + String(SILENCE_HANG_MS));
      meta.close();
    }

    resetPreBuffer();
    burstActive = false;
    lastLoudMs = millis();
    isRecording = true;
    xSemaphoreGive(sdMutex);
    Serial.println("[Node" + String(NODE_ID) + "] Listening for sound above threshold (adaptive noise floor: " +
                   String(noiseFloorRms, 2) + ")...");
    return true;
  }

  void stopRecording()
  {
    if (!isRecording)
      return;
    if (xSemaphoreTake(sdMutex, portMAX_DELAY) != pdTRUE)
      return;
    isRecording = false;
    if (burstActive)
    {
      writeWavHeader(recordingFile, wavDataBytes);
      recordingFile.close();
      burstActive = false;
    }
    File meta = SD.open(currentFolder + "/meta.txt", FILE_APPEND);
    if (meta)
    {
      meta.println("End_ms=" + String(millis()));
      meta.println("NoiseFloorRMS_atEnd=" + String(noiseFloorRms, 2));
      meta.close();
    }
    xSemaphoreGive(sdMutex);
    Serial.println("[Node" + String(NODE_ID) + "] Stopped");
  }

  // --- Audio Task (Core 0) ---
  // ฟังตลอด แต่เขียนลง SD เฉพาะตอนเสียงเกิน soundThreshold (trigger) จนกว่าจะเงียบติดต่อกันเกิน SILENCE_HANG_MS
  void recordAudioChunk()
  {
    if (!isRecording)
      return;
    size_t bytesRead = 0;
    if (i2s_read(I2S_NUM_0, i2sRaw, sizeof(i2sRaw), &bytesRead, portMAX_DELAY) != ESP_OK || bytesRead == 0)
      return;

    const size_t samplesRead = bytesRead / sizeof(i2sRaw[0]);
    float sumSq = 0.0f;
    for (size_t i = 0; i < samplesRead; i++)
    {
      int32_t s = (i2sRaw[i] >> 16) * micGain;
      pcmBuffer[i] = static_cast<int16_t>(constrain(s, -32768, 32767));
      sumSq += (float)pcmBuffer[i] * pcmBuffer[i];
    }

    float rms = sqrtf(sumSq / samplesRead);
    lastRms = rms;

    // Adaptive noise floor: asymmetric EMA ตามลงไว (เจอช่วงเงียบจริงระหว่าง chirp) ตามขึ้นช้ามาก
    // (กัน chirp เองดันพื้นเสียงขึ้นเรื่อยๆ) — รันตลอดเวลาที่ isRecording แทนการ calibrate ครั้งเดียวตอน start
    // เพราะภาคสนามจริงบังคับให้จิ้งหรีดเงียบตอน start ไม่ได้
    if (!noiseFloorInitialized)
    {
      noiseFloorRms = rms;
      noiseFloorInitialized = true;
    }
    else
    {
      float alpha = (rms < noiseFloorRms) ? NOISE_FLOOR_ALPHA_DOWN : NOISE_FLOOR_ALPHA_UP;
      noiseFloorRms += alpha * (rms - noiseFloorRms);
    }
    soundThreshold = noiseFloorRms * THRESHOLD_MULTIPLIER;

    bool loud = rms > soundThreshold;
    if (loud)
      lastLoudMs = millis();

    // AGC: ปรับ gain เฉพาะตอนกำลังอัด burst จริงเท่านั้น — ถ้าปรับตอน "ฟังเฉยๆ" ด้วย gain จะลอยขึ้นเรื่อยๆ
    // ตอนไม่มีเสียง (เพราะ RMS พื้นหลังต่ำกว่า AGC_TARGET_RMS เสมอ) ทำให้ soundThreshold ที่ adaptive ไว้ใช้ไม่ได้
    if (burstActive && rms > 50.0f)
    {
      float targetGain = micGain * (AGC_TARGET_RMS / rms);
      micGain = (int16_t)constrain(
          (int)(micGain + AGC_ALPHA * (targetGain - micGain)),
          (int)AGC_GAIN_MIN, (int)AGC_GAIN_MAX);
    }

    if (!burstActive && !loud)
    {
      pushPreBuffer(pcmBuffer, samplesRead); // ฟังรอ trigger เฉยๆ เก็บไว้ pre-roll ไม่เขียน SD
      return;
    }

    const size_t bytesToWrite = samplesRead * sizeof(pcmBuffer[0]);

    if (xSemaphoreTake(sdMutex, portMAX_DELAY) != pdTRUE)
      return;

    if (!burstActive)
    {
      if (!openNextAudioFile())
      {
        xSemaphoreGive(sdMutex);
        return;
      }
      flushPreBufferToFile(); // เทเสียงที่สะสมไว้ก่อน trigger ลงไฟล์ใหม่ กันเสียงต้น chirp หาย
      burstActive = true;
      Serial.println("[Node" + String(NODE_ID) + "] Triggered: " + recordingPath);
    }

    // SD card ทำ internal housekeeping เป็นครั้งคราว ทำให้ write เดี่ยวๆ พลาดชั่วคราวได้ (ไม่ใช่การ์ดเสียจริง) — retry ก่อนตัดจบ session
    bool writeOk = false;
    for (int attempt = 0; attempt < 3 && !writeOk; attempt++)
    {
      if (attempt > 0)
      {
        // write พลาด → close/reopen แบบ append บังคับให้ driver คำนวณตำแหน่ง/cluster ใหม่ แทนที่จะ retry
        // ตำแหน่งเดิมที่ state อาจค้างเพี้ยนอยู่ (ทดสอบแล้วว่า retry เฉยๆ ไม่ช่วยหลัง write แรกพัง)
        Serial.println("[Node" + String(NODE_ID) + "] SD write retry " + String(attempt) + " (reopen)");
        recordingFile.close();
        recordingFile = SD.open(recordingPath, FILE_APPEND);
        if (!recordingFile)
          break;
      }
      writeOk = recordingFile.write(reinterpret_cast<uint8_t *>(pcmBuffer), bytesToWrite) == bytesToWrite;
    }
    if (!writeOk)
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
      if (!openNextAudioFile())
      {
        burstActive = false;
        isRecording = false;
      }
    }
    else if (millis() - lastLoudMs > SILENCE_HANG_MS)
    {
      writeWavHeader(recordingFile, wavDataBytes);
      recordingFile.close();
      burstActive = false;
      Serial.println("[Node" + String(NODE_ID) + "] Silence " + String(SILENCE_HANG_MS) + "ms — burst closed");
    }

    xSemaphoreGive(sdMutex);
  }

  // Debug: เขียนข้อมูลปลอมลง SD รัวๆ จังหวะเดียวกับตอนอัดจริง แต่ไม่แตะ I2S เลย — ใช้แยกว่าปัญหาเกิดจาก
  // I2S ชนกับ SPI ตอนทำงานพร้อมกัน หรือ SD เขียนต่อเนื่องหนักๆ ไม่ไหวเองล้วนๆ (เปิดใช้ผ่าน TEST_SD_ONLY)
  constexpr bool TEST_SD_ONLY = false;
  void testSdOnlyTask(void *)
  {
    if (!sdReady)
    {
      Serial.println("[TEST] SD not ready, aborting test");
      vTaskDelete(NULL);
    }
    SD.remove("/sdtest.bin");
    File f = SD.open("/sdtest.bin", FILE_WRITE);
    if (!f)
    {
      Serial.println("[TEST] open failed");
      vTaskDelete(NULL);
    }

    memset(pcmBuffer, 0xAA, sizeof(pcmBuffer));
    uint32_t writes = 0, fails = 0, recovered = 0;
    uint32_t startMs = millis();
    while (millis() - startMs < 60000) // ทดสอบ 60 วิ
    {
      bool ok = f.write(reinterpret_cast<uint8_t *>(pcmBuffer), sizeof(pcmBuffer)) == sizeof(pcmBuffer);
      writes++;
      if (!ok)
      {
        fails++;
        // write พลาด → close/reopen แบบ append บังคับให้ driver คำนวณตำแหน่ง/cluster ใหม่ แล้วลองอีกที
        f.close();
        f = SD.open("/sdtest.bin", FILE_APPEND);
        bool ok2 = f && f.write(reinterpret_cast<uint8_t *>(pcmBuffer), sizeof(pcmBuffer)) == sizeof(pcmBuffer);
        if (ok2)
          recovered++;
        Serial.printf("[TEST] write #%u failed (fail count: %u, recovered: %s)\n", writes, fails, ok2 ? "yes" : "no");
      }
      vTaskDelay(46 / portTICK_PERIOD_MS); // เลียนแบบจังหวะ i2s chunk จริง (~1024 samples @ 22050Hz)
    }
    Serial.printf("[TEST] Recovered %u / %u fails via reopen\n", recovered, fails);
    f.close();
    Serial.printf("[TEST] Done: %u writes, %u fails\n", writes, fails);
    vTaskDelete(NULL);
  }

  void audioTask(void *)
  {
    for (;;)
    {
      recordAudioChunk();
      vTaskDelay(1);
    }
  }

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
    json += "\"recording\":" + String(isRecording ? "true" : "false") + ",";
    json += "\"bursting\":" + String(burstActive ? "true" : "false") + ",";
    json += "\"file\":\"" + recordingPath + "\",";
    json += "\"bytes\":" + String(wavDataBytes) + ",";
    json += "\"ip\":\"" + ip + "\",";
    json += "\"sdReady\":" + String(sdReady ? "true" : "false") + ",";
    json += "\"i2sReady\":" + String(i2sReady ? "true" : "false") + ",";
    json += "\"label\":\"" + recordingLabel + "\",";
    json += "\"stage\":\"" + currentStageLabel + "\",";
    json += "\"temp\":" + String(lastTemperature, 1) + ",";
    json += "\"humidity\":" + String(lastHumidity, 1) + ",";
    json += "\"micGain\":" + String(micGain) + ",";
    json += "\"noiseFloor\":" + String(noiseFloorRms, 1) + ",";
    json += "\"soundThreshold\":" + String(soundThreshold, 1) + ",";
    json += "\"node\":" + String(NODE_ID) + ",";
    json += "\"isMain\":" + String(IS_MAIN_NODE ? "true" : "false") + ",";
    json += "\"nodes\":" + String(secondaryCount + 1) + ",";
    json += "\"secondaries\":[";
    for (int i = 0; i < secondaryCount; i++)
    {
      if (i > 0)
        json += ",";
      json += "{\"node\":" + String(i + 1) +
              ",\"ip\":\"" + secondaryIPs[i] + "\"" +
              ",\"stage\":\"" + secondaryStage[i] + "\"" +
              ",\"temp\":" + String(secondaryTemp[i], 1) +
              ",\"humidity\":" + String(secondaryHumidity[i], 1) +
              ",\"recording\":" + String(secondaryRecording[i] ? "true" : "false") +
              ",\"sdReady\":" + String(secondarySdReady[i] ? "true" : "false") +
              "}";
    }
    json += "]}";
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", json);
  }

  // History สำหรับวาดกราฟ (~15 นาทีย้อนหลัง) — เรียงจากเก่าไปใหม่
  void handleHistory()
  {
    String json = "{\"temp\":[";
    size_t envStart = (envHistoryFilled < ENV_HISTORY_LEN) ? 0 : envHistoryIdx;
    for (size_t n = 0; n < envHistoryFilled; n++)
    {
      if (n > 0)
        json += ",";
      json += String(tempHistory[(envStart + n) % ENV_HISTORY_LEN], 1);
    }
    json += "],\"humidity\":[";
    for (size_t n = 0; n < envHistoryFilled; n++)
    {
      if (n > 0)
        json += ",";
      json += String(humidityHistory[(envStart + n) % ENV_HISTORY_LEN], 1);
    }
    json += "],\"rms\":[";
    size_t rmsStart = (rmsHistoryFilled < RMS_HISTORY_LEN) ? 0 : rmsHistoryIdx;
    for (size_t n = 0; n < rmsHistoryFilled; n++)
    {
      if (n > 0)
        json += ",";
      json += String(rmsHistory[(rmsStart + n) % RMS_HISTORY_LEN], 0);
    }
    json += "],\"envIntervalMs\":" + String(ENV_HISTORY_INTERVAL) +
            ",\"rmsIntervalMs\":" + String(RMS_HISTORY_INTERVAL) + "}";
    // เปิด CORS เพราะหน้า dashboard ของ main จะ fetch /history ของ secondary ข้าม IP โดยตรง (ไม่ผ่าน proxy)
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", json);
  }

  // Main node: start own recording then broadcast to secondaries
  void handleStart()
  {
    startRecording();
    if (IS_MAIN_NODE)
      broadcastToSecondaries("/sync/start");
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

  // Secondary nodes: called by main via broadcast — ใช้ currentStageLabel ของตัวเอง ไม่รับ label จาก main
  void handleSyncStart()
  {
    startRecording();
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
      server.send(400, "text/plain", "Missing ip or node");
      return;
    }
    String ip = server.arg("ip");
    int nid = server.arg("node").toInt();
    if (nid < 1 || nid > MAX_SECONDARY)
    {
      server.send(400, "text/plain", "Invalid node");
      return;
    }

    int slot = nid - 1;
    if (secondaryIPs[slot] != ip)
    {
      secondaryIPs[slot] = ip;
      if (slot >= secondaryCount)
        secondaryCount = slot + 1;
      Serial.println("Registered node" + String(nid) + " -> " + ip);
    }
    server.send(200, "text/plain", "OK");
  }

  // Main node: secondary push stage/temp/humidity ของตัวเองมาให้ periodically (ทุก DHT_INTERVAL)
  void handleNodeStatus()
  {
    if (!server.hasArg("node"))
    {
      server.send(400, "text/plain", "Missing node");
      return;
    }
    int nid = server.arg("node").toInt();
    if (nid < 1 || nid > MAX_SECONDARY)
    {
      server.send(400, "text/plain", "Invalid node");
      return;
    }

    int slot = nid - 1;
    if (server.hasArg("stage"))
      secondaryStage[slot] = sanitizeLabel(server.arg("stage"));
    if (server.hasArg("temp"))
      secondaryTemp[slot] = server.arg("temp").toFloat();
    if (server.hasArg("humidity"))
      secondaryHumidity[slot] = server.arg("humidity").toFloat();
    if (server.hasArg("recording"))
      secondaryRecording[slot] = server.arg("recording") == "1";
    if (server.hasArg("sdReady"))
      secondarySdReady[slot] = server.arg("sdReady") == "1";
    server.send(200, "text/plain", "OK");
  }

  // ทุก node: ดู/แก้ growth stage ของบ่อที่ node นี้ติดตั้งอยู่ — เก็บใน NVS ไม่ต้อง reflash
  void handleGetStage()
  {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", "{\"stage\":\"" + currentStageLabel + "\"}");
  }

  void handleSetStage()
  {
    if (!server.hasArg("value"))
    {
      server.send(400, "text/plain", "Missing value");
      return;
    }
    String value = sanitizeLabel(server.arg("value"));

    bool valid = false;
    for (size_t i = 0; i < STAGE_OPTIONS_COUNT; i++)
      if (value == STAGE_OPTIONS[i])
      {
        valid = true;
        break;
      }
    if (!valid)
    {
      server.send(400, "text/plain", "Unknown stage value");
      return;
    }

    currentStageLabel = value;
    stagePrefs.putString("stage", currentStageLabel);
    Serial.println("[Node" + String(NODE_ID) + "] Stage set to: " + currentStageLabel);
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", "{\"stage\":\"" + currentStageLabel + "\"}");
  }

  void handleFiles()
  {
    String json = "[";
    bool first = true;
    if (sdReady && xSemaphoreTake(sdMutex, portMAX_DELAY) == pdTRUE)
    {
      File root = SD.open("/");
      File folder = root.openNextFile();
      while (folder)
      {
        if (folder.isDirectory())
        {
          File file = folder.openNextFile();
          while (file)
          {
            if (!first)
              json += ",";
            json += "{\"name\":\"/" + String(folder.name()) + "/" + String(file.name()) + "\",\"size\":" + String(file.size()) + "}";
            first = false;
            file = folder.openNextFile();
          }
        }
        folder = root.openNextFile();
      }
      xSemaphoreGive(sdMutex);
    }
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", json + "]");
  }

  void handleDownload()
  {
    if (!server.hasArg("file"))
    {
      server.send(400, "text/plain", "Missing file param");
      return;
    }
    if (!sdReady)
    {
      server.send(503, "text/plain", "SD not mounted");
      return;
    }
    String path = server.arg("file");
    if (!path.startsWith("/"))
      path = "/" + path;
    if (xSemaphoreTake(sdMutex, portMAX_DELAY) == pdTRUE)
    {
      File f = SD.open(path, FILE_READ);
      if (!f)
      {
        xSemaphoreGive(sdMutex);
        server.send(404, "text/plain", "Not found");
        return;
      }
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
    main { width: min(1200px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0; }
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
    .tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
    .tabBtn { min-height: 40px; padding: 0 16px; border-radius: 8px; border: 1px solid #d8e0e8; background: #fff; color: #3e4b58; font-size: 0.9rem; }
    .tabBtn.active { background: #0b5cad; color: #fff; border-color: #0b5cad; }
    .tabPanel { display: none; }
    .tabPanel.active { display: block; }
    .groupHead { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
    .nodeStageCtl { font-size: 0.85rem; color: #516171; display: flex; align-items: center; gap: 6px; }
    .nodeStageCtl select { border-radius: 6px; border: 1px solid #c7d2d8; padding: 3px 6px; font-size: 0.85rem; }
    .nodeStageCtl button { min-height: 28px; padding: 0 10px; background: #0b5cad; color: #fff; font-size: 0.82rem; }
    .nodeStageMsg { font-weight: 700; color: #146c43; }
    .charts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    .charts .panel { margin-bottom: 0; }
    .chartHead { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }
    .legend { font-size: 0.82rem; color: #516171; display: flex; align-items: center; gap: 6px; }
    .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-left: 8px; }
    .legend i:first-child { margin-left: 0; }
    canvas { width: 100%; height: 160px; display: block; }
    ul { list-style: none; padding: 0; margin: 18px 0 0; display: grid; gap: 10px; }
    li { display: flex; align-items: center; justify-content: space-between; gap: 12px; border-top: 1px solid #e6ebf0; padding-top: 10px; }
    a  { color: #0b5cad; font-weight: 700; overflow-wrap: anywhere; }
    @media (max-width: 900px) {
      .charts { grid-template-columns: 1fr; }
    }
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
      .tabBtn { background: #1b2530; border-color: #344354; color: #bac7d5; }
      .tabBtn.active { background: #0b5cad; color: #fff; border-color: #0b5cad; }
      .legend { color: #bac7d5; }
      .nodeStageCtl { color: #bac7d5; }
      .nodeStageCtl select { background: #1b2530; color: #edf2f7; border-color: #344354; }
      .nodeStageMsg { color: #70d49b; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <h1>Cricket Audio Logger</h1>
  </header>

  <div class="tabs" id="tabBar">
    <button class="tabBtn active" type="button" data-tab="overview">Overview</button>
  </div>

  <section class="panel tabPanel active" data-tab="overview">
    <div class="meta">
      <div>Status: <span class="status" id="status">Loading...</span></div>
    </div>
    <div class="meta" id="nodesRow">
      <div>Nodes online: <span class="nodes" id="nodeList"></span></div>
    </div>
    <div class="controls" id="recordControls">
      <button class="record" id="record">Record All</button>
      <button class="stop"   id="stop">Stop All</button>
    </div>
    <div class="notice" id="notice"></div>
  </section>

  <div id="overviewCharts" class="tabPanel active" data-tab="overview"></div>

  <div id="nodeTabPanels"></div>
</main>
<script>
  const statusEl  = document.querySelector('#status');
  const recordControlsEl = document.querySelector('#recordControls');
  const nodesRowEl = document.querySelector('#nodesRow');
  const noticeEl  = document.querySelector('#notice');
  const recordBtn = document.querySelector('#record');
  const stopBtn   = document.querySelector('#stop');
  const nodeListEl= document.querySelector('#nodeList');
  const tabBarEl  = document.querySelector('#tabBar');
  const overviewChartsEl = document.querySelector('#overviewCharts');
  const nodeTabPanelsEl  = document.querySelector('#nodeTabPanels');
  let currentNodes = []; // [{key, shortLabel, label, ip}] — สร้างใหม่ทุกครั้งที่ renderStatus จาก /status ล่าสุด

  function showTab(tab) {
    document.querySelectorAll('.tabPanel').forEach(p => p.classList.toggle('active', p.dataset.tab === tab));
    document.querySelectorAll('.tabBtn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    if (tab !== 'overview') {
      const node = currentNodes.find(n => n.key === tab);
      if (node) refreshNodeFiles(node);
    }
  }
  tabBarEl.addEventListener('click', e => {
    const btn = e.target.closest('.tabBtn');
    if (btn) showTab(btn.dataset.tab);
  });

  async function api(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }

  async function fetchWithTimeout(url, ms) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), ms);
    try { return await fetch(url, { signal: ctrl.signal }); }
    finally { clearTimeout(t); }
  }

  function renderStatus(d) {
    statusEl.textContent  = d.recording ? (d.bursting ? 'Recording (sound detected)' : 'Listening...') : 'Idle';
    recordBtn.disabled    = d.recording;
    stopBtn.disabled      = !d.recording;
    noticeEl.textContent  = d.recording
      ? `Session active on ${d.nodes} node(s)...`
      : `Ready — ${d.nodes - 1} secondary node(s) connected`;

    recordControlsEl.style.display = d.isMain ? '' : 'none';
    nodesRowEl.style.display       = d.isMain ? '' : 'none';

    nodeListEl.innerHTML = '';
    const self = document.createElement('span');
    self.className = 'node-badge self';
    self.textContent = 'Main (' + d.ip + ') · Stage ' + d.stage + ' · ' + d.temp + '°C/' + d.humidity + '%' +
      ' · SD ' + (d.sdReady ? 'OK' : 'ERR');
    nodeListEl.appendChild(self);
    (d.secondaries || []).forEach(s => {
      const b = document.createElement('span');
      b.className = 'node-badge';
      b.textContent = 'Node' + s.node + ' (' + s.ip + ') · Stage ' + (s.stage || '-') +
        ' · ' + s.temp + '°C/' + s.humidity + '%' + (s.recording ? ' · REC' : '') +
        ' · SD ' + (s.sdReady ? 'OK' : 'ERR');
      nodeListEl.appendChild(b);
    });

    // สร้างรายชื่อ node — main เห็นทุก node (fetch /history, /status, /files ตรงจาก IP แต่ละตัว), secondary เห็นแค่ของตัวเอง
    currentNodes = [{
      key: 'n0',
      shortLabel: d.isMain ? 'Main' : 'This Node',
      label: (d.isMain ? 'Main' : 'This node') + ' (' + d.ip + ')',
      ip: d.ip
    }];
    if (d.isMain) {
      (d.secondaries || []).forEach(s => {
        currentNodes.push({
          key: 'n' + s.node,
          shortLabel: 'Node' + s.node,
          label: 'Node' + s.node + ' (' + s.ip + ')',
          ip: s.ip
        });
      });
    }
  }

  // จุดสุดท้ายในทุก history array คือ "ตอนนี้" เสมอ ส่วน interval ต่อจุดคงที่ (10s สำหรับ temp/humidity, 1s สำหรับ rms)
  // เลยคำนวณ timestamp ย้อนหลังของแต่ละจุดฝั่ง client ได้เลย ไม่ต้องให้ ESP32 ส่ง timestamp มาด้วย
  const ENV_HISTORY_INTERVAL_MS = 10000;
  const RMS_HISTORY_INTERVAL_MS = 1000;

  // แกน X นับขึ้นจาก 0 (จุดเก่าสุดในกราฟ) ไปจนถึงเวลาปัจจุบัน ไม่ใช่นับถอยหลังจาก "ตอนนี้"
  function formatElapsed(ms) {
    const total = Math.round(ms / 1000);
    return Math.floor(total / 60) + ':' + String(total % 60).padStart(2, '0');
  }

  // วาดกราฟเส้นเดี่ยวง่ายๆ ด้วย canvas ล้วน (หน้านี้ไม่มีทาง access CDN ได้ เพราะ ESP32 เป็น AP แยกจาก internet)
  // yPad = ขยายช่วงแกน Y ออกจาก min/max ของข้อมูลจริง กันเส้นแนบขอบเกินไป
  // intervalMs = ห่างกันกี่ ms ต่อจุด ใช้คำนวณ timestamp แกน X และ tooltip
  function drawLineChart(canvas, data, color, yPad, intervalMs) {
    // ปรับความละเอียด canvas ตามขนาดจริงบนจอ x devicePixelRatio กันตัวหนังสือ/เส้นเบลอบนจอ HiDPI
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // เว้นขอบไม่เท่ากันแต่ละด้าน: padL กว้างพอให้ label ค่าแกน Y (เช่น "38.1") วางชิดซ้ายเส้นแกนได้
    // padB เว้นให้ label เวลาแกน X วางใต้เส้นแกนได้ — ไม่ใช่ลอยอยู่มุมจอเหมือนก่อนหน้านี้
    const padL = 34, padR = 8, padT = 8, padB = 16;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    ctx.clearRect(0, 0, w, h);
    if (data.length < 2) { canvas._chart = null; return; }
    const min = Math.min(...data) - yPad, max = Math.max(...data) + yPad;
    const range = (max - min) || 1;

    // เส้นแกน X (ล่าง) และแกน Y (ซ้าย) ให้ดูเป็นกราฟจริง ไม่ใช่แค่เส้นข้อมูลลอยๆ
    ctx.strokeStyle = 'rgba(138,151,163,0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    data.forEach((v, i) => {
      const x = padL + (i / (data.length - 1)) * plotW;
      const y = padT + plotH - ((v - min) / range) * plotH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = '#8a97a3';
    ctx.font = '11px system-ui, sans-serif';
    // ค่าแกน Y — ชิดซ้ายเส้นแกน Y (เหมือน tick label ของกราฟจริง)
    ctx.textAlign = 'right';
    ctx.fillText(max.toFixed(1), padL - 4, padT + 9);
    ctx.fillText(min.toFixed(1), padL - 4, padT + plotH);
    // เวลาแกน X — นับขึ้นจาก 0 ที่จุดเก่าสุด (ซ้ายสุด) ไปจนถึงตอนนี้ (ขวาสุด)
    ctx.textAlign = 'left';
    const spanMs = (data.length - 1) * intervalMs;
    ctx.fillText(formatElapsed(0), padL, h - 4);
    ctx.textAlign = 'right';
    ctx.fillText(formatElapsed(spanMs), padL + plotW, h - 4);
    ctx.textAlign = 'left';

    canvas._chart = { data, padL, plotW, intervalMs }; // เก็บไว้ให้ tooltip ตอน hover คำนวณค่า+เวลาตรงตำแหน่งเมาส์
  }

  // Tooltip แสดงค่า+เวลาตรงตำแหน่งเมาส์บนกราฟ (delegate event เดียวคลุมทุก canvas เพราะสร้าง/ลบ dynamic ตลอด)
  const chartTooltipEl = document.createElement('div');
  chartTooltipEl.id = 'chartTooltip';
  chartTooltipEl.style.cssText = 'position:fixed;display:none;pointer-events:none;background:#18202a;color:#fff;' +
    'padding:4px 8px;border-radius:6px;font-size:0.8rem;font-weight:700;z-index:1000;white-space:nowrap;';
  document.body.appendChild(chartTooltipEl);

  function hideChartTooltip() { chartTooltipEl.style.display = 'none'; }

  function handleChartPointer(e) {
    const canvas = e.target.closest('canvas');
    if (!canvas || !canvas._chart || canvas._chart.data.length < 2) { hideChartTooltip(); return; }
    const { data, padL, plotW, intervalMs } = canvas._chart;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const frac = (x - padL) / plotW;
    const idx = Math.round(Math.max(0, Math.min(1, frac)) * (data.length - 1));
    const elapsedMs = idx * intervalMs;
    chartTooltipEl.textContent = data[idx].toFixed(1) + '  (' + formatElapsed(elapsedMs) + ')';
    chartTooltipEl.style.left = (e.clientX + 12) + 'px';
    chartTooltipEl.style.top  = (e.clientY + 12) + 'px';
    chartTooltipEl.style.display = 'block';
  }

  overviewChartsEl.addEventListener('pointermove', handleChartPointer);
  overviewChartsEl.addEventListener('pointerleave', hideChartTooltip);

  // Overview tab: กราฟของทุก node เรียงต่อกัน (label เฉยๆ ไม่มีปุ่มควบคุม)
  function ensureOverviewChartGroup(node) {
    let group = overviewChartsEl.querySelector('[data-key="' + node.key + '"]');
    if (group) return group;
    group = document.createElement('div');
    group.className = 'nodeChartGroup';
    group.dataset.key = node.key;
    group.innerHTML =
      '<div class="groupHead"><strong></strong></div>' +
      '<div class="charts">' +
        '<section class="panel"><div class="chartHead"><span>Temperature</span><span class="legend"><i style="background:#ca2f3d"></i>°C</span></div><canvas class="cTemp"></canvas></section>' +
        '<section class="panel"><div class="chartHead"><span>Humidity</span><span class="legend"><i style="background:#0b5cad"></i>%</span></div><canvas class="cHumidity"></canvas></section>' +
        '<section class="panel"><div class="chartHead"><span>Sound Level (RMS)</span><span class="legend"><i style="background:#146c43"></i>RMS</span></div><canvas class="cRms"></canvas></section>' +
      '</div>';
    overviewChartsEl.appendChild(group);
    return group;
  }

  // แท็บของแต่ละ node: รายละเอียด (ไฟล์ปัจจุบัน, bytes, IP, hardware), ปุ่มตั้ง stage, และรายการไฟล์ใน SD
  function ensureNodeDetailTab(node) {
    let panel = nodeTabPanelsEl.querySelector('[data-tab="' + node.key + '"]');
    if (panel) return panel;

    const btn = document.createElement('button');
    btn.className = 'tabBtn';
    btn.type = 'button';
    btn.dataset.tab = node.key;
    btn.textContent = node.shortLabel;
    tabBarEl.appendChild(btn);

    panel = document.createElement('section');
    panel.className = 'panel tabPanel';
    panel.dataset.tab = node.key;
    panel.dataset.stageLoaded = '0';
    panel.innerHTML =
      '<div class="groupHead"><strong></strong>' +
        '<span class="nodeStageCtl">Stage: <select class="nodeStageSel">' +
          '<option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option>' +
        '</select> <button class="nodeStageSave" type="button">Save</button> <span class="nodeStageMsg"></span></span>' +
      '</div>' +
      '<div class="meta">' +
        '<div>Current file: <strong class="nFile">-</strong></div>' +
        '<div>Recorded data: <strong class="nBytes">0</strong> bytes <i>(Auto-split at 100 MB)</i></div>' +
        '<div>Device IP: <strong class="nIp">-</strong></div>' +
        '<div>Hardware: <strong class="nHardware">Checking...</strong></div>' +
        '<div>Noise floor / trigger threshold (adaptive): <strong class="nNoiseFloor">-</strong></div>' +
      '</div>' +
      '<div style="margin-top:16px;"><strong>Files on this node</strong><ul class="nFiles"></ul></div>';

    const sel = panel.querySelector('.nodeStageSel');
    const btnSave = panel.querySelector('.nodeStageSave');
    const msg = panel.querySelector('.nodeStageMsg');
    btnSave.addEventListener('click', async () => {
      btnSave.disabled = true;
      msg.textContent = '';
      try {
        const r = await fetchWithTimeout('http://' + node.ip + '/stage/set?value=' + encodeURIComponent(sel.value), 3000);
        const d = await r.json();
        msg.textContent = 'Saved: ' + d.stage;
      } catch (e) {
        msg.textContent = 'Failed';
      } finally {
        btnSave.disabled = false;
      }
    });

    nodeTabPanelsEl.appendChild(panel);
    return panel;
  }

  // เรียกเฉพาะตอนสลับไปดูแท็บของ node นั้น (ไม่ต้อง poll ถี่ๆ เพราะรายการไฟล์ไม่ได้เปลี่ยนบ่อย)
  async function refreshNodeFiles(node) {
    const panel = nodeTabPanelsEl.querySelector('[data-tab="' + node.key + '"]');
    const ul = panel && panel.querySelector('.nFiles');
    if (!ul) return;
    try {
      const r = await fetchWithTimeout('http://' + node.ip + '/files', 3000);
      const files = await r.json();
      ul.innerHTML = '';
      if (!files.length) { ul.innerHTML = '<li>No files</li>'; return; }
      for (const f of files) {
        const li   = document.createElement('li');
        const link = document.createElement('a');
        link.href        = 'http://' + node.ip + '/download?file=' + encodeURIComponent(f.name);
        link.textContent = f.name;
        const size = document.createElement('span');
        size.textContent = f.size + ' bytes';
        li.append(link, size);
        ul.append(li);
      }
    } catch (e) { ul.innerHTML = '<li>Failed to load files</li>'; }
  }

  async function refreshAllNodes() {
    const keys = new Set(currentNodes.map(n => n.key));
    let activeTabRemoved = false;

    overviewChartsEl.querySelectorAll('.nodeChartGroup').forEach(g => {
      if (!keys.has(g.dataset.key)) g.remove(); // node หลุดไปแล้ว (secondary offline) เอากราฟเก่าออก
    });
    nodeTabPanelsEl.querySelectorAll('.tabPanel').forEach(p => {
      if (!keys.has(p.dataset.tab)) {
        if (p.classList.contains('active')) activeTabRemoved = true;
        p.remove();
      }
    });
    tabBarEl.querySelectorAll('.tabBtn[data-tab]:not([data-tab="overview"])').forEach(b => {
      if (!keys.has(b.dataset.tab)) b.remove();
    });
    if (activeTabRemoved) showTab('overview');

    for (const node of currentNodes) {
      const chartGroup = ensureOverviewChartGroup(node);
      chartGroup.querySelector('strong').textContent = node.label;
      try {
        const r = await fetchWithTimeout('http://' + node.ip + '/history', 3000);
        if (r.ok) {
          const hd = await r.json();
          drawLineChart(chartGroup.querySelector('.cTemp'), hd.temp, '#ca2f3d', 5, ENV_HISTORY_INTERVAL_MS);
          drawLineChart(chartGroup.querySelector('.cHumidity'), hd.humidity, '#0b5cad', 10, ENV_HISTORY_INTERVAL_MS);
          drawLineChart(chartGroup.querySelector('.cRms'), hd.rms, '#146c43', 0, RMS_HISTORY_INTERVAL_MS);
        }
      } catch (e) { /* node นี้ตอบช้า/หลุด — ข้ามไปก่อน รอบหน้าค่อยลองใหม่ */ }

      const detailPanel = ensureNodeDetailTab(node);
      detailPanel.querySelector('strong').textContent = node.label;
      try {
        const r = await fetchWithTimeout('http://' + node.ip + '/status', 3000);
        if (r.ok) {
          const s = await r.json();
          detailPanel.querySelector('.nFile').textContent = s.file || '-';
          detailPanel.querySelector('.nBytes').textContent = s.bytes || 0;
          detailPanel.querySelector('.nIp').textContent = s.ip || node.ip;
          detailPanel.querySelector('.nHardware').textContent =
            (s.sdReady ? 'SD OK' : 'SD ERR') + ' / ' + (s.i2sReady ? 'I2S OK' : 'I2S ERR');
          detailPanel.querySelector('.nNoiseFloor').textContent =
            (s.noiseFloor ?? 0).toFixed(1) + ' / ' + (s.soundThreshold ?? 0).toFixed(1);
          // ตั้งค่า dropdown แค่ครั้งแรก กันไม่ให้ค่าที่คนกำลังเลือกอยู่โดน poll ทับ
          if (detailPanel.dataset.stageLoaded === '0' && s.stage) {
            detailPanel.querySelector('.nodeStageSel').value = s.stage;
            detailPanel.dataset.stageLoaded = '1';
          }
        }
      } catch (e) { /* node นี้ตอบช้า/หลุด — ข้ามไปก่อน รอบหน้าค่อยลองใหม่ */ }
    }
  }

  async function refresh() {
    try { renderStatus(await api('/status')); await refreshAllNodes(); }
    catch (e) { noticeEl.textContent = e.message; }
  }

  recordBtn.addEventListener('click', async () => {
    renderStatus(await api('/start'));
  });
  stopBtn.addEventListener('click', async () => {
    renderStatus(await api('/stop'));
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
    // ทุก node มี endpoints เหล่านี้ — รวม "/" ด้วย เพราะทุก node ต้องตั้ง stage ของตัวเองผ่านหน้าเว็บได้
    // (ปุ่ม Record/Stop และ nodes list จะถูกซ่อนฝั่ง JS ถ้าไม่ใช่ main node)
    server.on("/", HTTP_GET, handleIndex);
    server.on("/status", HTTP_GET, sendJsonStatus);
    server.on("/history", HTTP_GET, handleHistory);
    server.on("/sync/start", HTTP_GET, handleSyncStart);
    server.on("/sync/stop", HTTP_GET, handleSyncStop);
    server.on("/stage", HTTP_GET, handleGetStage);
    server.on("/stage/set", HTTP_GET, handleSetStage);
    server.on("/files", HTTP_GET, handleFiles);
    server.on("/download", HTTP_GET, handleDownload);

    // Main node เพิ่ม endpoint คุมการอัดทั้งฟาร์ม + registration
    if (IS_MAIN_NODE)
    {
      server.on("/start", HTTP_GET, handleStart);
      server.on("/stop", HTTP_GET, handleStop);
      server.on("/register", HTTP_GET, handleRegister);
      server.on("/nodestatus", HTTP_GET, handleNodeStatus);
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
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 4,
      .dma_buf_len = (int)I2S_SAMPLES,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0};
  i2s_pin_config_t pins = {
      .bck_io_num = MIC_SCK_PIN,
      .ws_io_num = MIC_WS_PIN,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = MIC_SD_PIN};
  if (i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL) != ESP_OK)
    return false;
  if (i2s_set_pin(I2S_NUM_0, &pins) != ESP_OK)
    return false;
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
  if (code != 200)
  {
    http.end();
    return;
  }
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
    Serial.println("[Node" + String(NODE_ID) + "] Main is recording — joining with local stage: " + currentStageLabel);
    startRecording();
  }
}

// Secondary node: ส่ง stage/temp/humidity/recording/sdReady ปัจจุบันให้ main เก็บไว้โชว์บน dashboard
void pushStatusToMain()
{
  if (IS_MAIN_NODE || WiFi.status() != WL_CONNECTED)
    return;
  HTTPClient http;
  String url = "http://" + String(MAIN_NODE_IP) + "/nodestatus?node=" + String(NODE_ID) +
               "&stage=" + currentStageLabel +
               "&temp=" + String(lastTemperature, 1) +
               "&humidity=" + String(lastHumidity, 1) +
               "&recording=" + String(isRecording ? "1" : "0") +
               "&sdReady=" + String(sdReady ? "1" : "0");
  http.begin(url);
  http.setConnectTimeout(2000);
  http.setTimeout(2000);
  http.GET();
  http.end();
}

// Secondary node: แจ้ง IP และ NODE_ID ให้ main node ทราบ (retry 5 ครั้ง)
void registerWithMain()
{
  if (IS_MAIN_NODE || WiFi.status() != WL_CONNECTED)
    return;
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
  sdMutex = xSemaphoreCreateMutex();
  if (DHT_ENABLED)
    dht.begin();

  stagePrefs.begin("cricket", false);
  currentStageLabel = stagePrefs.getString("stage", DEFAULT_STAGE_LABEL);
  Serial.println("[Node" + String(NODE_ID) + "] Stage loaded: " + currentStageLabel);

  sdReady = SD.begin(SD_CS_PIN, SPI, 4000000); // กลับมาใช้ default 4MHz — ปัญหาสาย/การ์ดแก้ไปแล้ว 1MHz อาจกระทบจังหวะเขียนต่อเนื่อง
  if (sdReady)
  {
    uint8_t cardType = SD.cardType();
    const char *typeStr = cardType == CARD_MMC ? "MMC" : cardType == CARD_SD ? "SDSC"
                                                     : cardType == CARD_SDHC ? "SDHC"
                                                                             : "UNKNOWN";
    Serial.printf("[Node%d] SD card type: %s, size: %lluMB, used: %lluMB, free: %lluMB\n", NODE_ID, typeStr,
                  SD.cardSize() / (1024 * 1024), SD.usedBytes() / (1024 * 1024),
                  (SD.totalBytes() - SD.usedBytes()) / (1024 * 1024));
  }
  else
  {
    Serial.printf("[Node%d] SD.begin() failed — check wiring/CS pin/card seating\n", NODE_ID);
  }
  i2sReady = setupI2S();
  Serial.printf("[Node%d] SD:%s I2S:%s\n", NODE_ID, sdReady ? "OK" : "FAIL", i2sReady ? "OK" : "FAIL");

  if (!IS_MAIN_NODE)
  {
    WiFi.onEvent([](WiFiEvent_t, WiFiEventInfo_t)
                 { registerWithMain(); },
                 ARDUINO_EVENT_WIFI_STA_GOT_IP);
  }

  setupWiFi();
  registerWithMain(); // no-op on main node
  setupWebServer();
  if (TEST_SD_ONLY)
    xTaskCreatePinnedToCore(testSdOnlyTask, "SDTest", 8192, NULL, 2, NULL, 0);
  else
    xTaskCreatePinnedToCore(audioTask, "Audio", 8192, NULL, 2, NULL, 1); // core 1 กันชนกับ WiFi internal task ที่มักอยู่ core 0
}

void loop()
{
  server.handleClient();

  if (DHT_ENABLED && millis() - lastDhtReadTime >= DHT_INTERVAL)
  {
    lastDhtReadTime = millis();
    float h = dht.readHumidity();
    float t = dht.readTemperature();

    if (isnan(h) || isnan(t))
    {
      Serial.println("DHT read failed");
      return;
    }
    Serial.printf("[Node%d] DHT: %.1f C  %.1f %%\n", NODE_ID, t, h);
    lastTemperature = t;
    lastHumidity = h;

    if (DHT_SD_LOGGING && isRecording && sdReady && dhtLogPath.length() > 0 && xSemaphoreTake(sdMutex, portMAX_DELAY) == pdTRUE)
    {
      File f = SD.open(dhtLogPath, FILE_APPEND);
      if (f)
      {
        f.printf("%lu,%.2f,%.2f\n", millis(), t, h);
        f.close();
      }
      xSemaphoreGive(sdMutex);
    }

    pushStatusToMain(); // no-op บน main node
  }

  if (millis() - lastEnvHistoryPushMs >= ENV_HISTORY_INTERVAL)
  {
    lastEnvHistoryPushMs = millis();
    pushEnvHistory();
  }

  if (millis() - lastRmsHistoryPushMs >= RMS_HISTORY_INTERVAL)
  {
    lastRmsHistoryPushMs = millis();
    pushRmsHistory();
  }
}
