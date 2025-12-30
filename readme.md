# 🗣️ Piper TTS Forge 

A streamlined, automated pipeline for fine-tuning **[Piper TTS](https://github.com/rhasspy/piper)** models.

This repository contains a set of sequential scripts that handle the full voice-training lifecycle: from slicing raw audio and transcribing it with Whisper, to training via PyTorch, and finally exporting an optimized ONNX model for inference.

---

## 🚀 Features

- **Automated Dataset Prep**  
  Uses OpenAI Whisper for transcription and Librosa for audio slicing.

- **Zero-Config Training Defaults**  
  Tuned for *Medium* quality (22,050 Hz) to balance realism and speed.

- **Live Training Dashboard**  
  A real-time monitor (`5_dashboard.py`) that generates waveform previews and spectrograms during training.

- **Checkpoint Safety Net**  
  A checkpoint manager (`8_checkpoint_manager.py`) to back up and restore training states.

- **Instant Inference**  
  A CLI chat interface (`7_talk.py`) to test your voice model immediately.

---

## 🛠️ Prerequisites

1. **Python 3.10+**
2. **NVIDIA GPU** (strongly recommended — CPU training may take weeks)
3. **Piper Engine Binary** (download separately)

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/piper-voice-trainer.git
cd piper-voice-trainer
````

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Piper

* Download the appropriate binary from the
  👉 [https://github.com/rhasspy/piper/releases](https://github.com/rhasspy/piper/releases)
* Extract it so the `piper/` directory lives in the **project root**

---

## ⚙️ Configuration

Edit `config.py`:

```python
VOICE_NAME = "my_custom_voice"  # Name of your trained model
BATCH_SIZE = 8                 # Reduce to 4 or 2 if VRAM is limited
MAX_EPOCHS = 6000              # Total training duration
```

---

## 🏃 Workflow Overview

Follow the scripts **in order** (1 → 8).

---

### 🔹 Phase 1: Setup & Data

#### 1️⃣ Setup

```bash
python 1_setup.py
```

* Creates required folders
* Verifies base model presence
* Prompts you to download the base checkpoint (~400 MB) if missing

➡️ **Action:** Place your `.wav` or `.mp3` files into `raw_audio/`

---

#### 2️⃣ Slice & Transcribe

```bash
python 2_slice_and_transcribe.py
```

* Splits audio into sentence-level clips
* Transcribes using Whisper
* Generates `metadata.csv`

---

#### 3️⃣ Preprocess Dataset

```bash
python 3_preprocess.py
```

* Converts audio + metadata into Piper-ready tensors

---

### 🔹 Phase 2: Training

#### 4️⃣ Train the Model

```bash
python 4_train.py
```

* Resume-safe (restart anytime)
* Press **Ctrl+C** to stop without losing progress

---

#### 5️⃣ Live Dashboard (new terminal)

```bash
python 5_dashboard.py
```

* Watches training checkpoints
* Auto-generates `preview_progress.wav` on each save
* Edit `prompt.txt` to change preview text

---

### 🔹 Phase 3: Management & Export

#### 8️⃣ Checkpoint Manager

```bash
python 8_checkpoint_manager.py
```

Use frequently to:

* Back up training state
* Restore older checkpoints if quality degrades

---

#### 6️⃣ Export Model

```bash
python 6_export.py
```

* Selects best checkpoint
* Outputs `.onnx` model to `final_models/`

---

### 🔹 Phase 4: Usage

#### 7️⃣ Interactive CLI

```bash
python 7_talk.py
```

* Type text → press Enter
* Audio saved to `generated_wavs/`
* Filenames auto-generated from first five words

---

## 📂 Folder Structure

```
.
├── 1_setup.py
├── ... (scripts 1–8)
├── config.py
├── requirements.txt
├── piper/                  # Piper binary directory
├── raw_audio/              # Source recordings
├── dataset/                # Processed training data
├── training_checkpoints/   # PyTorch logs (large)
├── final_models/           # Exported ONNX models
└── generated_wavs/         # Output from 7_talk.py
```

---

## 🐛 Troubleshooting

**Out of Memory (OOM)**
→ Lower `BATCH_SIZE` in `config.py`

**Metallic / robotic voice**
→ Overtraining
→ Restore an earlier checkpoint using script `8`

**Piper not found**
→ Confirm `piper/` exists in the project root and contains the executable

---

## 📜 Credits

Built on the excellent work of **Rhasspy / Piper**.
