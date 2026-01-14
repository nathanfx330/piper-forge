import os
import sys
import time
import glob
import subprocess
import torch
import soundfile as sf
import librosa
import json
import re
from config import *

# --- SETTINGS ---
SEARCH_ROOT = "." 
PREVIEW_DIR = "vocoder_previews"
VOCODER_CODE_DIR = "BigVGAN"

# --- SETUP ---
if not os.path.exists(PREVIEW_DIR): os.makedirs(PREVIEW_DIR)

sys.path.append(os.path.abspath(VOCODER_CODE_DIR))
try:
    from bigvgan import BigVGAN
except ImportError:
    print("❌ Error: Could not find BigVGAN source code.")
    sys.exit(1)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_config(config_path):
    with open(config_path, "r") as f:
        data = json.load(f)
    return AttrDict(data)

def generate_piper_draft(text):
    piper_binary = os.path.join(PIPER_DIR, "piper")
    if sys.platform == "win32": piper_binary += ".exe"
    piper_model = os.path.join(OUTPUT_DIR, f"{VOICE_NAME}.onnx")
    
    draft_filename = os.path.join(PREVIEW_DIR, "draft_reference.wav")
    if os.path.exists(draft_filename): return draft_filename

    print(f"   🎤 Generating Piper Draft...")
    cmd = [piper_binary, "--model", piper_model, "--output_file", draft_filename]
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.communicate(input=text.encode('utf-8'))
        return draft_filename
    except Exception:
        return None

def get_mel(wav_path, hparams):
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

def render_preview(ckpt_path, draft_wav, config_path):
    try:
        h = get_config(config_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        generator = BigVGAN(h).to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        
        if 'generator' in state_dict:
            generator.load_state_dict(state_dict['generator'])
        else:
            generator.load_state_dict(state_dict)
            
        generator.remove_weight_norm()
        generator.eval()
        
        with torch.no_grad():
            mel = get_mel(draft_wav, h)
            audio = generator(mel)
            audio = audio.squeeze().cpu().numpy()
            
        step_name = os.path.basename(ckpt_path)
        out_name = os.path.join(PREVIEW_DIR, f"preview_{step_name}.wav")
        
        sf.write(out_name, audio, h.sampling_rate)
        print(f"   ✨ Rendered: {os.path.basename(out_name)}")
        return True
    except Exception as e:
        print(f"   ⚠️  Render failed for {os.path.basename(ckpt_path)}: {e}")
        return False

def find_all_checkpoints():
    """
    Finds files starting with 'g_' followed by numbers.
    Catches g_00000400 (no extension), g_00000400.pt, g_00000400.pth
    """
    candidates = []
    # Search recursively for anything starting with g_
    raw_files = glob.glob(os.path.join(SEARCH_ROOT, "**", "g_*"), recursive=True)
    
    for f in raw_files:
        filename = os.path.basename(f)
        # Regex: matches g_ followed by digits, optionally followed by extension
        if re.match(r"g_\d+(\.pt|\.pth)?$", filename):
            # Exclude the NVIDIA base model
            if "pretrained_checkpoints" in f: continue
            candidates.append(f)
            
    return sorted(candidates)

def main():
    print(f"--- 🔭 BigVGAN Dashboard (Extension-Agnostic Mode) ---")
    
    prompt_path = "prompt.txt"
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r') as f: text = f.read().strip()
    else: text = "This is a test."
        
    draft_wav = generate_piper_draft(text)
    if not draft_wav: 
        print("Could not generate draft.")
        return

    config_path = os.path.join(VOCODER_CODE_DIR, "configs", "finetune_22khz.json")
    processed = []

    print("\n   👀 Initial Scan...")
    checkpoints = find_all_checkpoints()
    
    if checkpoints:
        print(f"   ✅ Found {len(checkpoints)} existing checkpoints. Rendering...")
        for ckpt in checkpoints:
            print(f"   ▶️  Processing {os.path.basename(ckpt)}...")
            if render_preview(ckpt, draft_wav, config_path):
                processed.append(ckpt)
    else:
        print("   ℹ️  No checkpoints found yet. Waiting...")

    print("\n   👀 Entering Live Watch Mode...")
    
    while True:
        checkpoints = find_all_checkpoints()
        for ckpt in checkpoints:
            if ckpt not in processed:
                # Wait for write to finish
                if time.time() - os.path.getmtime(ckpt) > 2:
                    print(f"   🆕 New checkpoint detected: {ckpt}")
                    if render_preview(ckpt, draft_wav, config_path):
                        processed.append(ckpt)
        
        time.sleep(10)

if __name__ == "__main__":
    main()