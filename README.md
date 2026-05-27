# 🦗 Cricket Acoustic Perception

> **วิทยานิพนธ์ปริญญาโท** — การวิเคราะห์และจัดกลุ่มเสียงจิ้งหรีดในฟาร์มเชิงพาณิชย์  
> โดยใช้ Unsupervised Learning เพื่อตรวจวัดพฤติกรรม (ความหิว, อัตราการรอดชีวิต)

## ภาพรวมโครงการ

ระบบนี้วิเคราะห์ **Soundscape ของฟาร์มจิ้งหรีด** (เสียงหลายตัวซ้อนทับกัน) โดยไม่ต้องแยกเสียงรายตัว  
ใช้ pipeline: **Raw Audio → Feature Extraction → UMAP → HDBSCAN → Behavior Alert**

```mermaid
graph TD
    A([🎙️ Raw Audio .wav]) --> B[1️⃣ audio_utils\nLoad · Segment 5s · Denoise]
    B --> C[2️⃣ features\nMFCC · Spectral · Chroma\nRMS · ACI → 53-dim vector]
    C --> D[3️⃣ clustering\nUMAP — GPU cuML\nHDBSCAN — auto K]
    D --> E[4️⃣ behavior\nDolbear Law · Hunger Alert\nMortality Alert]
    E --> F1([⚠️ แจ้งเตือน\nควรให้อาหาร])
    E --> F2([🚨 แจ้งเตือน\nอัตราตายสูง])
    E --> F3([✅ ปกติ])

    style A fill:#1e1e3f,stroke:#7c85ff,color:#fff
    style B fill:#1a2a1a,stroke:#4caf50,color:#fff
    style C fill:#1a2a1a,stroke:#4caf50,color:#fff
    style D fill:#2a1a2a,stroke:#ce93d8,color:#fff
    style E fill:#2a1a1a,stroke:#ff8a65,color:#fff
    style F1 fill:#3a2a00,stroke:#ffc107,color:#fff
    style F2 fill:#3a0000,stroke:#f44336,color:#fff
    style F3 fill:#003a00,stroke:#66bb6a,color:#fff
```

## โครงสร้างโปรเจกต์

```
cricket_perception/
├── src/cricket_perception/
│   ├── audio_utils.py      # load, segment, denoise
│   ├── features.py         # MFCC, Spectral, RMS, ACI, FeatureExtractor
│   ├── clustering.py       # UMAP + HDBSCAN (GPU auto-detect via cuML)
│   └── behavior.py         # Dolbear's Law, hunger alert, mortality alert
├── notebooks/
│   ├── 01_explore_dataset.ipynb    # ดู waveform, mel-spectrogram
│   ├── 02_feature_extraction.ipynb # สกัด features → บันทึก .npy
│   └── 03_clustering.ipynb         # UMAP + HDBSCAN + scatter plot
├── scripts/
│   └── download_dataset.sh         # ดาวน์โหลด InsectSet32 จาก Zenodo
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

| Notebook | เนื้อหา |
|---|---|
| `01_explore_dataset` | ดู waveform, mel-spectrogram, species distribution |
| `02_feature_extraction` | สกัด MFCC+Spectral+RMS+ACI → บันทึกเป็น `.npy` |
| `03_clustering` | UMAP dimensionality reduction + HDBSCAN clustering |

### 5. รัน Tests

```bash
pytest tests/ -v
```

## Feature Pipeline

| Feature | Dim | บทบาท |
|---|---|---|
| MFCC (13 coeff × mean+std) | 26 | Timbre / texture ของเสียง |
| Spectral (centroid, bw, rolloff, entropy) | 8 | ความแหลม/ทุ้ม, ความซับซ้อน |
| Chroma + ZCR | 14 | Tonal content + chirp rate |
| RMS Energy | 4 | ความดังโดยรวม (mortality indicator) |
| Acoustic Complexity Index (ACI) | 1 | ความหลากหลายทางเสียง (biotic index) |
| **รวม** | **53** | |

## Behavior Analysis

| สัญญาณ | วิธีตรวจ | เกณฑ์ |
|---|---|---|
| **ความหิว** | Aggressive Song cluster fraction | > 35% ของ soundscape |
| **อัตราตาย** | RMS + ACI เทียบกับ baseline | < 40%/50% ของค่าปกติ |
| **Temperature** | Dolbear's Law Q10 correction | ปรับอัตโนมัติตามอุณหภูมิ |

## GPU Acceleration

ระบบ detect GPU อัตโนมัติ:

- **cuML (RAPIDS)** — GPU-accelerated UMAP + HDBSCAN (แนะนำ)
- **CPU fallback** — umap-learn + hdbscan (ทำงานได้เสมอ)

```bash
# ติดตั้ง cuML สำหรับ CUDA 12.x
pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12==24.10.*
```

## Research Notes

ดูรายละเอียดหลักการทางวิทยาศาสตร์ใน [`research_notes.md`](research_notes.md)

## License

MIT License — สำหรับการวิจัยเชิงวิชาการ