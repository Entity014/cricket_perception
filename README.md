# 🦗 Cricket Acoustic Perception

> **วิทยานิพนธ์ปริญญาโท** — การวิเคราะห์และจัดกลุ่มเสียงจิ้งหรีดในฟาร์มเชิงพาณิชย์  
> โดยใช้ Unsupervised Learning เพื่อตรวจวัดพฤติกรรม (ความหิว, อัตราการรอดชีวิต)

## ภาพรวมโครงการ

ระบบนี้วิเคราะห์ **Soundscape ของฟาร์มจิ้งหรีด** (เสียงหลายตัวซ้อนทับกัน) โดยไม่ต้องแยกเสียงรายตัว  
ใช้ pipeline: **Raw Audio → Feature Extraction → UMAP → HDBSCAN → Behavior Alert**

```
Raw Audio (.wav)
    │
    ├─ [1] audio_utils   load / segment (5s) / denoise
    ├─ [2] features      MFCC + Spectral + RMS + ACI  →  53-dim vector
    ├─ [3] clustering    UMAP (GPU) + HDBSCAN  →  cluster labels
    └─ [4] behavior      Dolbear's Law + hunger/mortality alerts
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