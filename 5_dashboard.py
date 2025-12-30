import os
import glob
import time
import datetime
import subprocess
import sys
import shutil
import re
from config import *

# Try importing visual libraries (optional)
try:
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_VISUALS = True
except ImportError:
    HAS_VISUALS = False
    print("⚠️  Tip: Run 'pip install matplotlib librosa numpy' to see voice spectrograms.")

# --- SETTINGS ---
PREVIEW_WAV = "preview_progress.wav"
PREVIEW_IMG = "preview_spectrogram.png"
PROMPT_FILE = "prompt.txt"
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")

# Epoch where the Base Model started (HFC Female is ~2868)
BASE_START_EPOCH = 2868 

# Colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def get_formatted_time(seconds):
    """Converts seconds into 00h:00m:00s format"""
    return str(datetime.timedelta(seconds=int(seconds)))

def get_reference_data():
    """Reads metadata.csv to find the first real clip and its text."""
    if not os.path.exists(METADATA_PATH):
        return None, None
    
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    if not lines: return None, None

    parts = lines[0].strip().split('|')
    if len(parts) < 2: return None, None
    
    filename = parts[0]
    text = parts[1]
    real_wav_path = os.path.join(DATASET_DIR, "wavs", filename)
    return real_wav_path, text

def check_training_health(current_epoch):
    added = current_epoch - BASE_START_EPOCH
    if added < 0: added = 0
    
    status = ""
    color = RESET
    
    if added < 500:
        status = "WARMUP (Stabilizing)"
        color = RED
    elif 500 <= added < 2000:
        status = "LEARNING (Getting Clearer)"
        color = YELLOW
    elif 2000 <= added < 4000:
        status = "SWEET SPOT (Ideal)"
        color = GREEN
    else:
        status = "RISK OF OVERFITTING (Check for metallic sound)"
        color = RED
        
    print(f"    Health: {color}{status} (+{added} local epochs){RESET}")

def generate_visuals(real_wav, ai_wav, epoch_name, is_synced):
    if not HAS_VISUALS: return
    
    try:
        y_real, sr_real = librosa.load(real_wav, sr=None)
        y_ai, sr_ai = librosa.load(ai_wav, sr=None)
        
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
        
        # Real
        D_real = librosa.amplitude_to_db(np.abs(librosa.stft(y_real)), ref=np.max)
        librosa.display.specshow(D_real, y_axis='log', x_axis='time', sr=sr_real, ax=ax[0])
        ax[0].set_title(f"REAL Target: {os.path.basename(real_wav)}")
        
        # AI
        D_ai = librosa.amplitude_to_db(np.abs(librosa.stft(y_ai)), ref=np.max)
        librosa.display.specshow(D_ai, y_axis='log', x_axis='time', sr=sr_ai, ax=ax[1])
        
        title = f"AI Checkpoint: {epoch_name}"
        if not is_synced:
            title += " (⚠️ Custom Text - Visuals Mismatched)"
            
        ax[1].set_title(title)
        
        plt.tight_layout()
        plt.savefig(PREVIEW_IMG)
        plt.close(fig)
        print(f"    🖼️  Visual comparison saved: {PREVIEW_IMG}")
    except Exception as e:
        print(f"Visualizer Error: {e}")

def main():
    print(f"--- 📡 Dashboard for '{VOICE_NAME}' ---")
    
    # 1. Load Reference Data (Fallback)
    real_wav, ref_text = get_reference_data()
    if not real_wav:
        print(f"❌ Error: Could not read {METADATA_PATH}")
        return

    print(f"📝 Synced Fallback: \"{ref_text[:30]}...\"")
    print(f"👀 Watching:        {TRAINING_DIR}")
    print(f"💡 Tip: Edit '{PROMPT_FILE}' to test custom words.")
    
    last_processed = ""
    last_ckpt_time = time.time()
    first_run = True
    
    piper_bin = os.path.join(PIPER_DIR, "piper")
    if sys.platform == "win32": piper_bin += ".exe"

    while True:
        search_pattern = os.path.join(TRAINING_DIR, "**", "*.ckpt")
        checkpoints = glob.glob(search_pattern, recursive=True)
        
        if checkpoints:
            newest = max(checkpoints, key=os.path.getmtime)
            
            if newest != last_processed:
                time.sleep(3) # Wait for file write
                
                # --- TIMER LOGIC ---
                current_time = time.time()
                time_diff = current_time - last_ckpt_time
                time_str = ""
                if not first_run:
                    time_str = f"⏱️  Time since last: {YELLOW}{get_formatted_time(time_diff)}{RESET}"
                else:
                    time_str = "⏱️  Timer Started..."
                
                filename = os.path.basename(newest)
                match = re.search(r"epoch=(\d+)", filename)
                epoch = int(match.group(1)) if match else 0
                
                print(f"\n{CYAN}[!] New Checkpoint: {filename}{RESET}")
                check_training_health(epoch)
                print(f"    {time_str}")
                
                # 1. Export ONNX (Temp)
                temp_onnx = "temp_dashboard.onnx"
                env = os.environ.copy()
                env["PYTHONPATH"] = os.path.join(PIPER_DIR, "src", "python")
                
                cmd_exp = [sys.executable, "-m", "piper_train.export_onnx", newest, temp_onnx]
                try:
                    subprocess.run(cmd_exp, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    shutil.copy(os.path.join(TRAINING_DIR, "config.json"), f"{temp_onnx}.json")
                    
                    # 2. DETERMINE TEXT TO SPEAK
                    text_to_speak = ref_text
                    is_synced = True
                    
                    if os.path.exists(PROMPT_FILE):
                        with open(PROMPT_FILE, 'r') as f:
                            user_text = f.read().strip()
                            if user_text:
                                text_to_speak = user_text
                                is_synced = False
                                print(f"    ⚠️  Reading custom prompt from {PROMPT_FILE}")

                    # 3. Synthesize
                    ps = subprocess.Popen(("echo", text_to_speak), stdout=subprocess.PIPE)
                    subprocess.check_output(
                        [piper_bin, "--model", temp_onnx, "--output_file", PREVIEW_WAV],
                        stdin=ps.stdout,
                        stderr=subprocess.DEVNULL
                    )
                    ps.wait()
                    print(f"    🗣️  Speaking: \"{text_to_speak[:40]}...\"")
                    print(f"    🎧 Saved to: {PREVIEW_WAV}")
                    
                    # 4. Visualize
                    generate_visuals(real_wav, PREVIEW_WAV, filename, is_synced)
                    
                except Exception as e:
                    print(f"    ❌ Error processing checkpoint: {e}")

                last_processed = newest
                last_ckpt_time = current_time
                first_run = False
        
        time.sleep(10)

if __name__ == "__main__":
    main()