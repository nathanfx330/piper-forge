🎙️ Piper TTS Forge

A streamlined, **batteries-included** toolkit for training custom Neural Text-to-Speech (TTS) voices using [Piper](https://github.com/rhasspy/piper).

This project automates the most painful parts of voice cloning:
- Automatic slicing and transcription using **OpenAI Whisper**
- Dataset formatting for Piper
- Training, checkpoint management, and export
- A real-time dashboard to *listen* to your model as it learns

---

## ⚠️ Hardware Warning & Performance (Important)

**Read this before running `2_slice_and_transcribe.py`.**

By default, the slicer uses the **Whisper “large” model** for maximum transcription accuracy.

- **VRAM requirement:** ~10 GB or more  
  (RTX 3080 / 4070 / better)

### If you have less VRAM (RTX 3060, 2060, GTX 1080, etc.)

You **must** switch Whisper to the **medium** model.

Edit `2_slice_and_transcribe.py`:

```python
# FROM:
model = whisper.load_model("large", device=device)

# TO:
model = whisper.load_model("medium", device=device)
````

The **medium** model:

* Uses far less VRAM
* Is much faster
* Is ~95% as accurate for clean speech

---

## 📂 Folder Structure

Before starting, your directory should look like this:

```text
.
├── piper/                 # Download from Piper GitHub and extract here
│   ├── piper              # Piper executable
│   └── src/               # Piper Python source
├── raw_audio/             # Put your long .wav / .mp3 files here
├── 1_setup.py
├── 2_slice_and_transcribe.py
├── 3_preprocess.py
├── 4_train.py
├── 5_dashboard.py
├── 6_export.py
├── 7_talk.py
├── 8_checkpoint_manager.py
├── config.py              # <-- EDIT THIS FIRST
└── environment.yml
```

---

## 🛠️ Prerequisites

### System Dependencies

**Windows**

* Visual Studio C++ Build Tools
* eSpeak-NG (must be in PATH)

**Linux (Ubuntu / Debian)**

```bash
sudo apt-get install espeak-ng g++
```

### Python Environment (Recommended: Conda)

```bash
conda env create -f environment.yml
conda activate piper-trainer
```

---

## 🚀 Usage Guide

### 1. Configuration & Setup

Open `config.py` and set your `VOICE_NAME`.

Then run:

```bash
python 1_setup.py
```

If the base model is missing, the script will:

* Print a download link
* Tell you exactly where to place it
* Require it to be named `base_model.ckpt`

---

### 2. Prepare Audio

Drop your recordings into the `raw_audio/` folder.

**Recommended:**

* Format: WAV, MP3, FLAC, M4A
* Length: 15–60 minutes total
* Single speaker only
* No music, no effects, minimal background noise

---

### 3. Slicing & Transcription

This script:

* Splits audio on silence
* Transcribes speech with Whisper
* Builds `metadata.csv`

```bash
python 2_slice_and_transcribe.py
```

Afterwards, quickly inspect:

```text
dataset/metadata.csv
```

If you see junk lines (e.g. “Copyright”, “Subtitle”),
delete them before continuing.

---

### 4. Preprocessing

Converts audio and text into Piper-ready tensors.

```bash
python 3_preprocess.py
```

---

### 5. Training

Start the training loop:

```bash
python 4_train.py
```

* Press **Ctrl+C** to pause safely
* Run the script again to resume
* Typical good voices emerge between **1000–4000 epochs**

---

### 6. Dashboard (Live Monitoring)

While training runs in one terminal, open another and run:

```bash
python 5_dashboard.py
```

The dashboard will:

* Detect new checkpoints automatically
* Speak a sample line every time one is saved
* Show training health (Warmup → Sweet Spot → Overfitting)
* Optionally generate spectrogram comparisons

---

### 7. Export Final Model

When the voice sounds right (usually during “Sweet Spot”):

```bash
python 6_export.py
```

Your final files will appear in:

```text
final_models/
├── your_voice.onnx
└── your_voice.onnx.json
```

---

### 8. Talk (Inference)

Test your new voice interactively:

```bash
python 7_talk.py
```

Audio will be saved automatically as `.wav` files.

---

## 🔧 Troubleshooting

### CUDA Out of Memory (Training)

Lower `BATCH_SIZE` in `config.py`.
Try: `16 → 8 → 4`

---

### “Piper source code not found”

Ensure your `piper/` directory contains a `src/` folder.
You likely downloaded the wrong Piper release.

---

### Voice sounds metallic or robotic

You likely **overfitted**:

* Too many epochs
* Too little data

Use:

```bash
python 8_checkpoint_manager.py
```

to restore an earlier checkpoint, or stop training sooner next time.

---

## ⚖️ License

This automation toolkit is open source.

The Piper engine itself is licensed under MIT
(c) Rhasspy contributors.

```

---

If you want, I can also:
- Add diagrams (training flow / data flow)
- Tighten it for public release vs personal use
- Write a short “When to stop training” guide
- Or split this into **Quick Start** + **Deep Dive**

Just say which direction.
```
