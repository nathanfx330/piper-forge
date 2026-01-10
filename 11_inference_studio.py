import os
import sys
import subprocess
import torch
import soundfile as sf
import librosa
import json
import numpy as np
import re
import glob
from config import *

# --- CONFIGURATION ---
VOCODER_DIR = "BigVGAN"
# Leave empty to auto-find your latest trained model
BIGVGAN_CKPT = "" 

# Piper Setup
PIPER_MODEL = os.path.join(OUTPUT_DIR, f"{VOICE_NAME}.onnx")
PIPER_BINARY = os.path.join(PIPER_DIR, "piper")
if sys.platform == "win32": PIPER_BINARY += ".exe"

# --- SETUP IMPORTS ---
sys.path.append(os.path.abspath(VOCODER_DIR))

try:
    from bigvgan import BigVGAN
except ImportError:
    print("❌ Error: Could not import BigVGAN.")
    sys.exit(1)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_config(config_path):
    with open(config_path, "r") as f:
        data = json.load(f)
    return AttrDict(data)

def get_mel_spectrogram(wav_path, hparams):
    wav, sr = librosa.load(wav_path, sr=hparams.sampling_rate, mono=True)
    wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
    spec = torch.stft(
        wav_tensor, n_fft=hparams.n_fft, hop_length=hparams.hop_length,
        win_length=hparams.win_length, window=torch.hann_window(hparams.win_length),
        center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex=True
    )
    spec = torch.abs(spec)
    mel_filter = librosa.filters.mel(
        sr=hparams.sampling_rate, n_fft=hparams.n_fft, n_mels=hparams.num_mels,
        fmin=hparams.fmin, fmax=hparams.fmax
    )
    mel_filter = torch.from_numpy(mel_filter).float()
    mel = torch.matmul(mel_filter, spec)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    if torch.cuda.is_available(): return mel.cuda()
    return mel

def load_bigvgan(checkpoint_path, config_path):
    print(f"   🧠 Loading Vocoder Weights...")
    hparams = get_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = BigVGAN(hparams).to(device)
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    
    if 'generator' in checkpoint_dict:
        model.load_state_dict(checkpoint_dict['generator'])
    elif 'model' in checkpoint_dict:
        model.load_state_dict(checkpoint_dict['model'])
    else:
        model.load_state_dict(checkpoint_dict)
        
    model.eval()
    model.remove_weight_norm()
    return model, hparams

def find_checkpoint():
    """Finds the newest g_xxxx file"""
    candidates = []
    for root, dirs, files in os.walk("."):
        if "/." in root: continue
        for file in files:
            if file.startswith("g_") and "pretrained" not in root:
                candidates.append(os.path.join(root, file))
    
    if not candidates: return None
    # Return newest
    return max(candidates, key=os.path.getmtime)

def main():
    print(f"--- 🎙️  Studio Inference (Custom Model) ---")
    
    # 1. Check Piper
    if not os.path.exists(PIPER_MODEL):
        print(f"❌ Error: Piper model not found at {PIPER_MODEL}")
        return

    # 2. Check BigVGAN (MOVED TO TOP)
    final_ckpt = BIGVGAN_CKPT if BIGVGAN_CKPT else find_checkpoint()
    
    if not final_ckpt:
        print("\n❌ Error: No trained checkpoints found.")
        print("   Did you run Script 10 to at least Step 200?")
        return

    # 3. Print Confirmation (MOVED TO TOP)
    ckpt_name = os.path.basename(final_ckpt)
    match = re.search(r"g_(\d+)", ckpt_name)
    step_num = match.group(1) if match else "???"
    
    print(f"   ✅ Auto-Selected Checkpoint: {ckpt_name}")
    print(f"      (Training Step: {step_num})")
    print("-" * 40)

    # 4. NOW ask for input
    text = input("📝 Text to Speak: ").strip()
    if not text: return

    draft_wav = "temp_draft.wav"
    print("\n   1️⃣  Piper: Generating acoustic draft...")
    
    cmd = [
        PIPER_BINARY, "--model", PIPER_MODEL, "--output_file", draft_wav,
        "--noise_scale", "0.667", "--length_scale", "1.0", "--noise_w", "0.8"
    ]
    
    try:
        subprocess.run(cmd, input=text.encode('utf-8'), check=True, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"❌ Piper Execution Failed: {e}")
        return

    # Load & Render
    config_path = os.path.join(VOCODER_DIR, "configs", "finetune_22khz.json")
    
    try:
        vocoder, hparams = load_bigvgan(final_ckpt, config_path)
        
        print("   2️⃣  BigVGAN: Rendering high-fidelity audio...")
        with torch.no_grad():
            mel = get_mel_spectrogram(draft_wav, hparams)
            audio = vocoder(mel)
            audio = audio.squeeze().cpu().numpy()
            
        out_filename = "studio_output.wav"
        sf.write(out_filename, audio, hparams.sampling_rate)
        
        print(f"\n✅ Done! Studio Master saved to: {out_filename}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if "Out of memory" in str(e):
            print("   👉 YOUR GPU IS FULL. Please close other scripts.")

if __name__ == "__main__":
    main()
