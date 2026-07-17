# 🦗 Cricket Acoustic Perception

> **วิทยานิพนธ์ปริญญาโท** — การวิเคราะห์เสียงจิ้งหรีดในฟาร์มเชิงพาณิชย์เพื่อจำแนก **ระยะการเจริญเติบโต (Growth Stage)**  
> โดยใช้ Supervised Learning จาก label ของผู้เชี่ยวชาญ ควบคู่กับการตรวจวัดพฤติกรรม (ความหิว, อัตราการรอดชีวิต)

## 🔄 สถานะโครงการ (Project Direction)

โครงการกำลังปรับทิศทางจาก **Unsupervised Behavior Clustering** (Phase เดิม) ไปสู่ **Supervised Growth-Stage Classification** (Phase ใหม่) เนื่องจากตอนนี้มีผู้เชี่ยวชาญ (expert) ที่สามารถระบุระยะการเจริญเติบโตของจิ้งหรีดได้จริง ทำให้ไม่ต้องพึ่งการ label จาก unsupervised cluster อีกต่อไป

**Growth Stage classes (ร่างเบื้องต้น — รอยืนยันชื่อ/นิยามจากผู้เชี่ยวชาญ):**

| # | Stage (ร่าง) | ลักษณะ |
| - | ------------ | ------ |
| 1 | ตัวอ่อน / เด็ก (Nymph) | ไม่มีปีก ตัวเล็ก |
| 2 | วัยรุ่น / หนุ่ม (Sub-adult) | ปีกเริ่มงอก (wing pad) |
| 3 | เพิ่งลอกคราบเป็นตัวเต็มวัย (Young adult) | ปีกสมบูรณ์แต่ยังนิ่ม |
| 4 | ตัวเต็มวัยพร้อมผสมพันธุ์ (Mature adult) | ปีกแข็งเต็มที่ พร้อมเก็บเกี่ยว/ผสมพันธุ์ |

**สถานะข้อมูล:** ยังไม่มี dataset ที่ label ไว้ — อยู่ระหว่างวางแผนเก็บข้อมูลใหม่ โดยใช้ฟีเจอร์ `Label` ที่มีอยู่แล้วในระบบ firmware ([Web UI](#web-ui)) เพื่อบันทึกเสียงแยกตามระยะที่ผู้เชี่ยวชาญยืนยันต่อ session

**ข้อมูลฟาร์มที่ยืนยันแล้ว:**
- ฟาร์มเลี้ยง**จิ้งหรีดสายพันธุ์เดียว** (ยังไม่ทราบว่าเป็นสายพันธุ์อะไร)
- โครงสร้างการเลี้ยงเป็น **1 บ่อ = 1 stage** (แต่ละบ่อมีจิ้งหรีดระยะเดียวกันทั้งหมด ไม่ปนกันหลาย stage ในบ่อเดียว) — ทำให้การ label เสียงต่อ session ทำได้ตรงไปตรงมา เพียงตั้งค่า `Label` ให้ตรงกับ stage ของบ่อที่ไปอัดเสียง ไม่ต้องกังวลเรื่องเสียงหลาย stage ปนกันในไฟล์เดียว
- มี **4 stage** จริง (ยืนยันจำนวนแล้ว แต่ยังไม่รู้ชื่อ/นิยามของแต่ละ stage) และ**ไม่นับไข่ (Egg) เป็นหนึ่งใน stage** — เริ่มนับจากฟักเป็นตัวอ่อน ดูรายละเอียดคำถามที่ยังค้างอยู่ที่ [docs/farm_visit_questions.md](docs/farm_visit_questions.md)

**แผนขั้นตอน (Growth-Stage Pipeline):**

1. **เก็บข้อมูลเสียง** แยกตาม session พร้อม label ระยะจากผู้เชี่ยวชาญ (ผ่าน firmware `Label` field) — คาดว่าจะได้ label ไม่ครบทุก segment ในช่วงแรก
2. **Unsupervised clustering (UMAP + HDBSCAN)** เป็น exploratory/validation step — รันบนข้อมูลทั้งหมด (labeled + unlabeled) เพื่อดูว่าเสียงแบ่งกลุ่มตาม acoustic signature ได้กี่กลุ่ม แล้วเทียบกับ label ที่มีจากผู้เชี่ยวชาญว่า cluster ↔ stage สอดคล้องกันแค่ไหน (ไม่ใช้ cluster แทน ground-truth label โดยตรง เพราะ cluster อาจแยกตาม noise/environment มากกว่า stage จริง โดยเฉพาะ stage 1-2 ที่แทบไม่มีเสียง chirp)
3. **Semi-supervised label propagation** — ถ้า cluster จับคู่กับ label ได้ดี ใช้ cluster ช่วย propagate label ไปยัง segment ที่ยังไม่มี label จากผู้เชี่ยวชาญ เพื่อประหยัดเวลา labeling
4. **Train supervised `GrowthStageClassifier`** บนชุดข้อมูลที่ label ครบ (จาก expert + propagated) — สถาปัตยกรรมเดิมจาก `SongTypeClassifier` (SVM/RF) นำมาปรับใช้ได้

**หมายเหตุ:** ระบบ `behavior.py` (hunger/mortality alert จาก Dolbear's Law + RMS/ACI baseline) ยังคงเก็บไว้แยกต่างหากตามเดิม ไม่ได้ถูกแทนที่โดย growth-stage classification — ทั้งสองระบบทำงานคู่ขนานกัน

## ภาพรวมโครงการ

ระบบนี้วิเคราะห์ **Soundscape ของฟาร์มจิ้งหรีด** (เสียงหลายตัวซ้อนทับกัน) โดยไม่ต้องแยกเสียงรายตัว

- **Growth-Stage Pipeline (ใหม่):** **Raw Audio → Feature Extraction → Supervised Classifier (label จากผู้เชี่ยวชาญ) → Growth Stage**
- **Behavior Alert Pipeline (เดิม, คงไว้):** **Raw Audio → Feature Extraction → UMAP → HDBSCAN → Behavior Alert**

> Diagram ด้านล่างคือ Behavior Alert Pipeline (Phase เดิม) ที่ยังคงทำงานอยู่ — diagram ของ Growth-Stage Pipeline (Phase ใหม่) จะถูกเพิ่มเข้ามาเมื่อมี dataset ที่ label แล้ว

```mermaid
flowchart TD
    subgraph Offline Phase ["1️⃣ Offline Training Phase (เทรนและตั้งค่าระบบ)"]
        A1["🎙️ ไฟล์เสียงตัวอย่างจากฟาร์ม/Dataset"] --> A2["audio_utils: Segment 5s & Denoise"]
        A2 --> A3["features: สกัด 53-dim Feature Vector"]
        A3 --> A4["clustering: UMAP + HDBSCAN"]
        A4 --> A5["notebook 05: จัดหมวดหมู่เสียงของแต่ละกลุ่มปุ่มเสียง\n- Calling Song / Aggressive / Quiet / Noise -"]
        A5 --> A6["classifier: เทรน SongTypeClassifier\n- SVM หรือ Random Forest -"]
        A6 --> A7["💾 Save: song_classifier.pkl"]

        A3 --> B1["notebook 06: วิเคราะห์สัญญาณเสียงบ่อปกติเพื่อหาเกณฑ์อ้างอิง\n- RMS / ACI Baseline -"]
        B1 --> B2["💾 Save: calibrated_thresholds.json"]
    end

    subgraph Online Phase ["2️⃣ Online Real-Time Phase (ตรวจจับและแจ้งเตือนเรียลไทม์)"]
        C1["🎙️ ไมโครโฟนสดในฟาร์ม หรือ ไฟล์เสียงยาวยี่สิบสี่ชั่วโมง"] --> C2["audio_utils: หั่นหน้าต่างเสียงทีละ 2.5s เลื่อนทุก 1.0s"]
        C2 --> C3["features: สกัด 53-dim Feature Vector\n- ~40-60ms -"]

        C3 --> C4["classifier: โหลด song_classifier.pkl เพื่อทำนายเสียง\n- <1ms -"]
        C4 --> C5{"ผลลัพธ์เสียงที่ทำนายได้"}

        C5 -->|Calling / Aggressive / Courtship| C6["behavior: BehaviorMonitor\n- โหลดค่าอ้างอิงจาก calibrated_thresholds.json\n- คำนวณปรับเกณฑ์กับอุณหภูมิปัจจุบัน -"]
        C5 -->|Quiet / Noise| C7["❌ ข้ามช่วงเวลานี้\n- ไม่แจ้งเตือนใดๆ -"]

        C6 --> C8["วิเคราะห์และประเมินผลลัพธ์"]
        C8 -->|Aggressive Song >= 35%| D1["⚠️ Hunger Alert\n- แจ้งเตือนความหิว -"]
        C8 -->|RMS หรือ ACI ต่ำกว่าปกติมากๆ| D2["🚨 Mortality Alert\n- แจ้งเตือนอัตราการตายสูง -"]
        C8 -->|สัญญาณเสียงเป็นปกติ| D3["✅ Normal\n- ฟาร์มอยู่ในสถานะปกติ -"]

        D1 & D2 --> E1(["📡 ส่งแจ้งเตือน\n- LINE Notify / MQTT / API -"])
    end

    style Offline Phase fill:#111122,stroke:#7c85ff,color:#fff
    style Online Phase fill:#112211,stroke:#69f0ae,color:#fff
```

## โครงสร้างโปรเจกต์

```
cricket_perception/
├── src/cricket_perception/
│   ├── audio_utils.py      # load, segment, denoise
│   ├── features.py         # MFCC, Spectral, RMS, ACI, FeatureExtractor
│   ├── clustering.py       # UMAP + HDBSCAN (GPU auto-detect via cuML)
│   ├── classifier.py       # SVM & RandomForest song-type classifier
│   ├── behavior.py         # Dolbear's Law, hunger alert, mortality alert
│   └── realtime.py         # Live mic and file streaming monitor
├── notebooks/
│   ├── 01_explore_dataset.ipynb    # ดู waveform, mel-spectrogram
│   ├── 02_feature_extraction.ipynb # สกัด features → บันทึก .npy
│   ├── 03_clustering.ipynb         # UMAP + HDBSCAN + scatter plot
│   ├── 04_behavior_analysis.ipynb   # จำลองการคำนวณ behavior alert
│   ├── 05_cluster_audio_labeling.ipynb # แปะป้ายจัดประเภทให้กลุ่มเสียง
│   ├── 06_baseline_calibration_protocol.ipynb # ตั้งค่าและ Calibrate บ่อ
│   └── 07_realtime_classifier.ipynb  # เทรนโมเดลและรัน Real-time demo
├── scripts/
│   ├── download_dataset.sh         # ดาวน์โหลด InsectSet32 จาก Zenodo
│   └── realtime_monitor.py         # สคริปต์สตรีมเสียงจริงจากไมค์
├── tests/
│   └── test_features.py            # Unit tests (21/21 passed)
├── requirements.txt
└── pyproject.toml
```

## เริ่มต้นใช้งาน

### 1. ติดตั้ง Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. ติดตั้ง PyTorch (GPU)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. ดาวน์โหลด Dataset

```bash
./scripts/download_dataset.sh
```

ดาวน์โหลด **InsectSet32** (Orthoptera) จาก [Zenodo record 7072196](https://zenodo.org/records/7072196)  
— 9 species, 147 WAV files (CC BY 4.0)

### 4. รัน Notebooks

```bash
jupyter notebook notebooks/
```

| Notebook                           | เนื้อหา                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------- |
| `01_explore_dataset`               | ดู waveform, mel-spectrogram, species distribution                              |
| `02_feature_extraction`            | สกัด MFCC+Spectral+RMS+ACI → บันทึกเป็น `.npy`                                  |
| `03_clustering`                    | UMAP dimensionality reduction + HDBSCAN clustering                              |
| `04_behavior_analysis`             | จำลองการคำนวณพฤติกรรมและการแจ้งเตือน (Dolbear's Law, hunger, mortality)         |
| `05_cluster_audio_labeling`        | ฟังเสียงตัวอย่างและระบุความหมายเสียงให้กับแต่ละ Cluster                         |
| `06_baseline_calibration_protocol` | ตั้งค่าและคำนวณเกณฑ์ค่าพลังงานอ้างอิงปกติ (Baseline) ของบ่อจิ้งหรีดที่สุขภาพดี  |
| `07_realtime_classifier`           | เทรนโมเดลแยกแยะประเภทเสียง (SVM/RF) และรันระบบตรวจจับแจ้งเตือนแบบเรียลไทม์จำลอง |

### 5. รัน Tests

```bash
pytest tests/ -v
```

## Feature Pipeline

> Audio recorded at **22 050 Hz** (firmware) → feature extraction ที่ `sr=22050` (Python) ตรงกัน

| Feature                                   | Dim    | บทบาท                               |
| ----------------------------------------- | ------ | ----------------------------------- |
| MFCC (13 coeff × mean+std)                | 26     | Timbre / texture ของเสียง           |
| Spectral (centroid, bw, rolloff, entropy) | 8      | ความแหลม/ทุ้ม, ความซับซ้อน          |
| Chroma + ZCR                              | 14     | Tonal content + chirp rate          |
| RMS Energy                                | 4      | ความดังโดยรวม (mortality indicator) |
| Acoustic Complexity Index (ACI)           | 1      | ความหลากหลายทางเสียง (biotic index) |
| **รวม**                                   | **53** |                                     |

## Behavior Analysis

| สัญญาณ          | วิธีตรวจ                         | เกณฑ์                                              |
| --------------- | -------------------------------- | -------------------------------------------------- |
| **ความหิว**     | Aggressive Song cluster fraction | calibrated จาก feeding trial (default 35%)          |
| **อัตราตาย**    | RMS + ACI เทียบกับ baseline      | calibrated จาก mortality trial (default 40%/50%)   |
| **Temperature** | Dolbear's Law Q10 correction     | ปรับอัตโนมัติตามอุณหภูมิ                           |
| **Circadian**   | cosine weighting ตามเวลากลางคืน  | ผ่อนปรน mortality threshold ตอนกลางวัน (0.1×–1.0×) |

### Threshold Calibration จาก Feeding Trial จริง

```python
from cricket_perception.behavior import BehaviorMonitor

monitor = BehaviorMonitor(
    rms_baseline=0.05, aci_baseline=250.0,
    aggressive_cluster_ids=[2], calling_cluster_ids=[0],
)

# trial_records = list of {"aggressive_frac": ..., "rms_ratio": ..., "aci_ratio": ...}
# was_hungry = [True, False, True, ...]  ← บันทึกจากฟาร์มว่าช่วงไหนหิวจริง
report = monitor.calibrate_from_trials(
    records=trial_records,
    hunger_labels=was_hungry,
    mortality_labels=had_mortality,
)
print(report)
# {"hunger": {"threshold": 0.28, "auc": 0.87, "sensitivity": 0.82, "specificity": 0.79}, ...}
```

Youden's J statistic จาก ROC curve ให้ threshold ที่ maximize (sensitivity + specificity − 1)
พร้อม AUC สำหรับรายงานในวิทยานิพนธ์

## Hardware

| Component     | รายละเอียด                                                                 |
| ------------- | -------------------------------------------------------------------------- |
| MCU           | ESP32 (dual-core, FreeRTOS)                                                |
| Microphone    | I2S MEMS mic — WS: GPIO42, SCK: GPIO41, SD: GPIO40                        |
| Sample Rate   | **22 050 Hz** (ตรงกับ Python pipeline), 16-bit mono WAV                   |
| Gain          | Software AGC — target RMS ≈ 2 500 (−22 dBFS), range ×1–×16, EMA α = 0.05 |
| Sensor        | DHT22 (GPIO4) — Temperature + Humidity ทุก 10 วินาที                      |
| Storage       | SD card (SPI, CS: GPIO10), auto-split ที่ 100 MB/ไฟล์                     |
| Multi-node    | WiFi AP/STA — Main node เปิด AP, Secondary node join และ sync อัตโนมัติ    |
| Power         | Built-in battery — millis() ต่อเนื่อง ไม่ reset                           |

### Folder structure บน SD card

```
/node0_D1_Food_D2_Food_12345/
    meta.txt          # Node ID, Label, Start_ms, End_ms
    audio_0.wav       # ไฟล์เสียง (≤100 MB ต่อไฟล์)
    data_0.csv        # Timestamp_ms, Temperature_C, Humidity_pct
    audio_1.wav       # auto-split ถัดไป
    data_1.csv
```

### Web UI

Main node เปิด AP `Cricket-Audio` (password `12345678`) — เข้า `http://192.168.4.1`

- เลือก Label (สภาวะการทดลอง) → กด **Record All** → sync ไปทุก node พร้อมกัน
- ดู status realtime: SD, I2S, Mic Gain (AGC), Nodes online
- Download ไฟล์โดยตรงจาก browser

## Domain Gap & Augmentation

ระบบนี้มาพร้อมกับฟีเจอร์การทำ **Mixture Augmentation** เพื่อแก้ปัญหาความแตกต่างระหว่างข้อมูลชุดฝึกสอน (เสียงเดี่ยว/สะอาด) และเสียงจริงในฟาร์ม (เสียงจิ้งหรีดร้องซับซ้อนทับซ้อนกัน):

- **SoundscapeSynthesizer**: สุ่มผสมสัญญาณเสียงจิ้งหรีดหลายๆ ตัวพร้อมปรับระดับความดัง (Gain) และการเหลื่อมของเวลา (Jitter) เพื่อจำลองความหนาแน่นและระยะห่างของจิ้งหรีด
- **Noise Injection**: เพิ่มเสียงรบกวนสังเคราะห์ (Pink noise, White noise, Brown noise) ที่ระดับ SNR ต่างๆ เพื่อจำลองเสียงสภาพแวดล้อมที่แท้จริง
- **Composition-based Labeling**: วิเคราะห์พลังงานเสียงผสมเพื่อระบุคลาสเสียง (เช่น หากสัดส่วนเสียง Aggressive Song ในกลุ่มเสียงหลักมีพลังงานรวมเกิน 30% จะกำหนดเป็นคลาส "Aggressive Song" เพื่อใช้กระตุ้นสัญญาณเตือนภัย)

ตัวอย่างการฝึกฝนโมเดลร่วมกับฟีเจอร์สังเคราะห์ข้อมูล:

```python
from cricket_perception.classifier import SongTypeClassifier
from cricket_perception.features import FeatureExtractor

# เตรียม FeatureExtractor
extractor = FeatureExtractor(sr=22050)

# เทรนโมเดลร่วมกับเทคนิค Augmentation
clf = SongTypeClassifier(backend="svm")
metrics = clf.train_with_augmentation(
    X_original=X_train,
    y_original=y_train,
    segments_csv="results/segments_with_clusters.csv",
    labels_csv="results/04_cluster_labels.csv",
    dataset_dir="dataset/insectset32/Orthoptera/Orthoptera",
    extractor=extractor,
    aug_ratio=0.5, # สร้างข้อมูลสังเคราะห์ 50% ของขนาดข้อมูลจริง
    cv_folds=5,
)
print("Augmented Training Complete:", metrics)
```

## GPU Acceleration

ระบบ detect GPU อัตโนมัติ:

- **cuML (RAPIDS)** — GPU-accelerated UMAP + HDBSCAN (แนะนำ)
- **CPU fallback** — umap-learn + hdbscan (ทำงานได้เสมอ)

```bash
# ติดตั้ง cuML สำหรับ CUDA 12.x
pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12==24.10.*
```

## Streaming Long Recordings

Pipeline รองรับไฟล์เสียงยาวหลายชั่วโมงโดยไม่ต้องโหลดทั้งไฟล์เข้า RAM:

### Memory-efficient I/O

```python
from cricket_perception.audio_utils import stream_audio

# ไฟล์ 24 ชม. ใช้ RAM ~3 MB แทน ~7.2 GB
for window, t in stream_audio("farm_24h.wav", window_sec=2.5, hop_sec=1.0):
    features = extractor.extract(window)
```

### Real-time Monitor + CSV Logging

```python
from cricket_perception.realtime import RealTimeMonitor

monitor = RealTimeMonitor(classifier_path="results/song_classifier.pkl")
for result in monitor.stream_file(
    "farm_24h.wav",
    csv_path="results/farm_2026-05-29.csv",   # บันทึกผลทุก window เป็น CSV
    aggregate_minutes=5.0,                     # smoothing ทุก 5 นาที
):
    if result.alerts:
        print(result.alerts)
```

### Temporal Smoothing

```python
from cricket_perception.realtime import RollingAggregator

agg = RollingAggregator(window_minutes=5.0)
for result in monitor.stream_file("farm.wav"):
    agg.add(result)
    summary = agg.summary()
    # summary["smoothed_hungry"]         — ดูจากค่าเฉลี่ย 5 นาที (ลด false positive)
    # summary["smoothed_mortality_risk"] — majority vote จาก 5 นาที
    # summary["song_type_fractions"]     — สัดส่วนเสียงแต่ละประเภทใน 5 นาที
```

| Component | หน้าที่ |
|-----------|---------|
| `stream_audio()` | อ่านไฟล์ทีละ block (30s) + yield sliding window → RAM คงที่ |
| `RollingAggregator` | Smooth ผลลัพธ์ข้ามเวลา N นาที → ลด false positive |
| `ResultLogger` | เขียน CSV ทีละแถว → วิเคราะห์ time-series ย้อนหลังได้ |

## License

MIT License — สำหรับการวิจัยเชิงวิชาการ
