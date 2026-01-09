import os
import sys
import subprocess
import glob
import torch
import soundfile as sf
import librosa
import numpy as np
from config import *

# --- CONFIGURATION ---

# 1. Path to the BigVGAN repo folder
VOCODER_DIR = "BigVGAN"

# 2. Your Fine-Tuned Checkpoint (Update this filename after training!)
#    If this specific file isn't found, the script will try to auto-pick the newest one.
BIGVGAN_CKPT = "vocoder_checkpoints/g_00010000" 

# 3. Piper Model (The "Draftsman")
#    We use the finalized ONNX model from Script 6
PIPER_MODEL = os.path.join(OUTPUT_DIR, f"{VOICE_NAME}.onnx")
PIPER_BINARY = os.path.join(PIPER_DIR, "piper")
if sys.platform == "win32": PIPER_BINARY += ".exe"

# --- SETUP IMPORTS ---
# We need to import modules from the BigVGAN repository itself
sys.path.append(os.path.abspath(VOCODER_DIR))

try:
    from models import BigVGAN
    from utils import get_hparams_from_file
except ImportError:
    print("❌ Error: Could not import BigVGAN modules.")
    print(f"   Ensure the folder '{VOCODER_DIR}' exists in this directory.")
    sys.exit(1)

def get_mel_spectrogram(wav_path, hparams):
    """
    Converts audio to Mel Spectrogram using EXACTLY the same math
    as the BigVGAN training loop. This ensures the Vocoder understands the input.
    """
    # 1. Load Audio
    # Force loading at 22050Hz Mono to match training specs
    wav, sr = librosa.load(wav_path, sr=hparams.data.sampling_rate, mono=True)
    
    # 2. Convert to Tensor
    wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
    
    # 3. Compute STFT (Short-Time Fourier Transform)
    # This extracts the frequency "Blueprint" from the audio
    spec = torch.stft(
        wav_tensor,
        n_fft=hparams.data.filter_length,
        hop_length=hparams.data.hop_length,
        win_length=hparams.data.win_length,
        window=torch.hann_window(hparams.data.win_length),
        center=True,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True
    )
    spec = torch.abs(spec)
    
    # 4. Convert to Mel Scale
    # This compresses the data to match human hearing perception
    mel_filter = librosa.filters.mel(
        sr=hparams.data.sampling_rate,
        n_fft=hparams.data.filter_length,
        n_mels=hparams.data.n_mel_channels,
        fmin=hparams.data.mel_fmin,
        fmax=hparams.data.mel_fmax
    )
    mel_filter = torch.from_numpy(mel_filter).float()
    
    mel = torch.matmul(mel_filter, spec)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    
    if torch.cuda.is_available():
        return mel.cuda()
    return mel

def load_bigvgan(checkpoint_path, config_path):
    print(f"   🧠 Loading BigVGAN Brain: {os.path.basename(checkpoint_path)}")
    
    # Load parameters from our fine-tune config
    hparams = get_hparams_from_file(config_path)
    
    # Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BigVGAN(hparams).to(device)
    
    # Load Weights
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats (raw vs state_dict)
    if 'model' in checkpoint_dict:
        model.load_state_dict(checkpoint_dict['model'])
    else:
        model.load_state_dict(checkpoint_dict)
        
    model.eval()
    # Removing weight norm makes inference faster
    model.remove_weight_norm()
    return model, hparams

def main():
    print(f"--- 🎙️  Studio Inference (Piper + BigVGAN) ---")
    
    # 0. Check Piper Setup
    if not os.path.exists(PIPER_MODEL):
        print(f"❌ Error: Piper model not found at {PIPER_MODEL}")
        print("   Run '6_export.py' first.")
        return

    # 1. Input Prompt
    text = input("\n📝 Text to Speak: ").strip()
    if not text: return

    # 2. Generate "Draft" with Piper
    draft_wav = "temp_draft.wav"
    print("\n   1️⃣  Piper: Generating acoustic draft (structure & prosody)...")
    
    cmd = [
        PIPER_BINARY,
        "--model", PIPER_MODEL,
        "--output_file", draft_wav
    ]
    
    try:
        # Run Piper process
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.communicate(input=text.encode('utf-8'))
    except Exception as e:
        print(f"❌ Piper Execution Failed: {e}")
        return
    
    if not os.path.exists(draft_wav):
        print("❌ Piper failed to generate audio file.")
        return

    # 3. Locate & Load Vocoder
    config_path = os.path.join(VOCODER_DIR, "configs", "finetune_22khz.json")
    
    # Logic to find the best checkpoint
    final_ckpt = BIGVGAN_CKPT
    if not os.path.exists(final_ckpt):
        # Auto-find the newest checkpoint if the configured one is missing
        search_path = os.path.join("vocoder_checkpoints", "g_*.pt")
        ckpts = glob.glob(search_path)
        if ckpts:
            final_ckpt = max(ckpts, key=os.path.getctime)
            print(f"   ⚠️  Configured checkpoint not found. Using newest found: {os.path.basename(final_ckpt)}")
        else:
            print("❌ No BigVGAN checkpoints found in 'vocoder_checkpoints/'.")
            print("   Have you run '10_train_vocoder.py' yet?")
            return

    vocoder, hparams = load_bigvgan(final_ckpt, config_path)

    # 4. Resynthesize (The Bridge)
    print("   2️⃣  BigVGAN: Rendering high-fidelity audio (texture & physics)...")
    
    try:
        with torch.no_grad():
            # A. Audio -> Mel (Analysis)
            mel = get_mel_spectrogram(draft_wav, hparams)
            
            # B. Mel -> High-Res Audio (Synthesis)
            audio = vocoder(mel)
            
            # C. Save
            audio = audio.squeeze().cpu().numpy()
            
        out_filename = "studio_output.wav"
        sf.write(out_filename, audio, hparams.data.sampling_rate)
        
        print(f"\n✅ Done! Studio Master saved to: {out_filename}")
        print("   Compare the two files:")
        print(f"   - Draft:  {draft_wav} (Piper Native)")
        print(f"   - Master: {out_filename} (Neural Vocoder)")

    except Exception as e:
        print(f"\n❌ Error during rendering: {e}")
        if "CUDA" in str(e):
            print("   (This might be an Out of Memory error if your text is very long)")

if __name__ == "__main__":
    main()
