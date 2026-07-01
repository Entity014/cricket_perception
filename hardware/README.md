# Hardware — Cricket Audio Logger

ระบบบันทึกเสียงจิ้งหรีดแบบ multi-node บน ESP32-S3 พร้อมการบันทึกสภาพแวดล้อม (อุณหภูมิ/ความชื้น) ลง microSD และ Web UI สำหรับควบคุมผ่าน WiFi

---

## สารบัญ

- [ภาพรวม](#ภาพรวม)
- [อุปกรณ์ที่ต้องใช้](#อุปกรณ์ที่ต้องใช้)
- [วงจรการเชื่อมต่อ](#วงจรการเชื่อมต่อ)
- [โครงสร้างโปรเจกต์](#โครงสร้างโปรเจกต์)
- [การติดตั้งและ Flash](#การติดตั้งและ-flash)
- [การใช้งาน](#การใช้งาน)
- [โครงสร้างไฟล์บน SD Card](#โครงสร้างไฟล์บน-sd-card)
- [Web API](#web-api)
- [การตั้งค่าเพิ่มเติม](#การตั้งค่าเพิ่มเติม)

---

## ภาพรวม

ระบบประกอบด้วย ESP32-S3 สูงสุด **3 โหนด** (1 Main + 2 Secondary) ทำงานพร้อมกัน:

- **Main node** สร้าง WiFi Access Point และมี Web UI สำหรับ Start/Stop การบันทึกพร้อมกันทุกโหนด
- **Secondary node** เชื่อมต่อ WiFi ของ Main และรับคำสั่งอัตโนมัติ
- แต่ละโหนดบันทึก **เสียง 16 kHz mono WAV** และ **ข้อมูล DHT22 (temp/humidity)** ลง microSD ของตัวเอง
- ไฟล์เสียงถูก split อัตโนมัติทุก 100 MB

---

## อุปกรณ์ที่ต้องใช้

| อุปกรณ์ | รายละเอียด |
|---|---|
| ESP32-S3-DevKitC-1 | บอร์ดหลัก (ต่อโหนด) |
| MEMS Microphone (I2S) | เช่น INMP441 หรือ SPH0645 |
| MicroSD Card Module (SPI) | พร้อม microSD card |
| DHT22 | เซ็นเซอร์อุณหภูมิ/ความชื้น |
| Resistor 10kΩ | Pull-up สำหรับ DHT22 |

---

## วงจรการเชื่อมต่อ

### I2S Microphone

| Mic Pin | ESP32-S3 GPIO |
|---|---|
| WS (LRCK) | GPIO 42 |
| SCK (BCLK) | GPIO 41 |
| SD (DATA) | GPIO 40 |
| VDD | 3.3V |
| GND | GND |
| L/R | GND (mono left channel) |

### MicroSD (SPI)

| SD Module Pin | ESP32-S3 GPIO |
|---|---|
| CS | GPIO 10 |
| MOSI | GPIO 11 (SPI default) |
| SCK | GPIO 12 (SPI default) |
| MISO | GPIO 13 (SPI default) |
| VCC | 3.3V |
| GND | GND |

### DHT22

| DHT22 Pin | ESP32-S3 GPIO |
|---|---|
| DATA | GPIO 4 |
| VCC | 3.3V |
| GND | GND |

> ต่อ resistor 10kΩ ระหว่าง VCC และ DATA pin ของ DHT22

---

## โครงสร้างโปรเจกต์

```
hardware/
├── README.md
└── firmware/
    ├── platformio.ini       # Build config (ESP32-S3, Arduino framework)
    ├── src/
    │   └── main.cpp         # Firmware หลัก
    ├── include/
    └── lib/
```

**Dependencies (จัดการโดย PlatformIO อัตโนมัติ):**
- `adafruit/DHT sensor library @ ^1.4.6`
- `adafruit/Adafruit Unified Sensor @ ^1.1.14`

---

## การติดตั้งและ Flash

### ข้อกำหนด

- [PlatformIO](https://platformio.org/) (VS Code extension หรือ CLI)
- USB cable (USB-C)

### ขั้นตอน

**1. ตั้งค่า Node ID ก่อน flash แต่ละบอร์ด**

เปิด [firmware/src/main.cpp](firmware/src/main.cpp) แก้ไขบรรทัดต่อไปนี้:

```cpp
constexpr bool IS_MAIN_NODE = true;  // true = main, false = secondary
constexpr int  NODE_ID      = 0;     // 0 = main, 1 หรือ 2 = secondary
```

| โหนด | `IS_MAIN_NODE` | `NODE_ID` |
|---|---|---|
| Main | `true` | `0` |
| Secondary 1 | `false` | `1` |
| Secondary 2 | `false` | `2` |

**2. Build และ Upload**

```bash
cd hardware/firmware
pio run --target upload
```

หรือใช้ปุ่ม **Upload** ใน VS Code PlatformIO sidebar

**3. ตรวจสอบ Serial Monitor**

```bash
pio device monitor --baud 115200
```

ข้อความที่ควรเห็น (Main node):
```
[Node0] SD:OK I2S:OK
AP started — http://192.168.4.1
```

---

## การใช้งาน

### เริ่มต้นระบบ

1. Flash Main node → Secondary node(s) ตามลำดับ
2. เปิดไฟ Secondary node(s) ก่อน แล้วจึงเปิด Main node
3. Secondary จะเชื่อมต่อ WiFi `Cricket-Audio` (password: `12345678`) และลงทะเบียนกับ Main อัตโนมัติ
4. เชื่อมต่อ WiFi `Cricket-Audio` ด้วย laptop/phone แล้วเปิด `http://192.168.4.1`

### Web UI

หน้าเว็บแสดง:
- จำนวนโหนดที่ online และ IP address
- สถานะ SD card และ I2S microphone
- ขนาดไฟล์ที่บันทึกปัจจุบัน
- รายการไฟล์บน SD card พร้อม download link

**การบันทึก:**
1. เลือก **Label** ตามเงื่อนไขการทดลอง
2. กด **Record All** — Main จะส่งคำสั่งไปยัง Secondary พร้อมกัน
3. กด **Stop All** เมื่อต้องการหยุด

**Labels ที่มี:**

| Label | ความหมาย |
|---|---|
| `D1_Food_D2_Food` | วันที่ 1 ให้อาหาร / วันที่ 2 ให้อาหาร |
| `D1_NoFood_D2_Food` | วันที่ 1 ไม่ให้อาหาร / วันที่ 2 ให้อาหาร |
| `D1_Food_D2_NoFood` | วันที่ 1 ให้อาหาร / วันที่ 2 ไม่ให้อาหาร |
| `D1_NoFood_D2_NoFood` | วันที่ 1 ไม่ให้อาหาร / วันที่ 2 ไม่ให้อาหาร |

---

## โครงสร้างไฟล์บน SD Card

```
/
└── node0_D1_Food_D2_Food_123456/   ← session folder (nodeID_label_millis)
    ├── meta.txt                     ← Node ID, label, timestamp เริ่ม/สิ้นสุด
    ├── audio_0.wav                  ← เสียง 16kHz, 16-bit, mono PCM
    ├── data_0.csv                   ← DHT22 log คู่กับ audio_0
    ├── audio_1.wav                  ← ไฟล์ถัดไป (auto-split ที่ 100 MB)
    └── data_1.csv
```

**meta.txt ตัวอย่าง:**
```
Node=0
Label=D1_Food_D2_Food
Start_ms=12345
End_ms=99999
```

**data_N.csv format:**
```csv
Timestamp_ms,Temperature_C,Humidity_pct
12345,28.50,72.30
22345,28.60,72.10
```

> DHT22 อ่านค่าทุก **10 วินาที** และบันทึกเฉพาะช่วงที่กำลัง recording

---

## Web API

ทุกโหนดมี endpoints:

| Method | Endpoint | คำอธิบาย |
|---|---|---|
| GET | `/status` | JSON สถานะโหนด |
| GET | `/files` | JSON รายการไฟล์ใน SD |
| GET | `/download?file=<path>` | ดาวน์โหลดไฟล์ |
| GET | `/sync/start?label=<label>` | สั่ง start (ใช้โดย Main) |
| GET | `/sync/stop` | สั่ง stop (ใช้โดย Main) |

Main node เพิ่ม:

| Method | Endpoint | คำอธิบาย |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/start?label=<label>` | เริ่มบันทึกทุกโหนด |
| GET | `/stop` | หยุดบันทึกทุกโหนด |
| GET | `/register?ip=<ip>` | Secondary ลงทะเบียน IP |

**ตัวอย่าง `/status` response:**
```json
{
  "recording": true,
  "file": "/node0_D1_Food_D2_Food_12345/audio_0.wav",
  "bytes": 5242880,
  "ip": "192.168.4.1",
  "sdReady": true,
  "i2sReady": true,
  "label": "D1_Food_D2_Food",
  "node": 0,
  "nodes": 3,
  "secondaries": ["192.168.4.2", "192.168.4.3"]
}
```

---

## การตั้งค่าเพิ่มเติม

ค่าคงที่ที่ปรับได้ใน [firmware/src/main.cpp](firmware/src/main.cpp):

| ค่า | Default | ความหมาย |
|---|---|---|
| `SAMPLE_RATE` | `16000` | Sample rate (Hz) |
| `MIC_GAIN` | `4` | ขยายสัญญาณไมค์ |
| `MAX_FILE_SIZE` | `100 MB` | ขนาดสูงสุดก่อน split |
| `DHT_INTERVAL` | `10000 ms` | ความถี่อ่าน DHT22 |
| `MAX_SECONDARY` | `2` | จำนวน secondary node สูงสุด |
| `AP_SSID` | `Cricket-Audio` | ชื่อ WiFi |
| `AP_PASSWORD` | `12345678` | รหัส WiFi |
