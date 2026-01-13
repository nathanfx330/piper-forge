import os
import sys
import subprocess
import json
import random
import threading
import time
import glob
import math
from config import *

# --- ⚙️  VOCODER SETTINGS ---
# Change this number to fit your GPU VRAM
# 4 = Safe (100% stable)
# 6 = Aggressive (Good for 10GB-12GB VRAM)
# 8 = High (Requires 12GB+ free or 16GB VRAM)
TARGET_BATCH_SIZE = 6

# --- 🔍 SYSTEM CHECK & FEEDBACK ---
def print_status_report():
    print(f"\n{'-'*60}")
    print(f"📊 SYSTEM STATUS REPORT")
    print(f"{'-'*60}")
    
    # 1. Check Piper Context
    print(f"🔹 PIPER CONTEXT:")
    print(f"   • Voice Name:    {VOICE_NAME}")
    print(f"   • Target Quality: {QUALITY.upper()}")
    print(f"   • Dataset Folder: dataset/wavs/")
    
    # 2. Check BigVGAN Config Strategy
    print(f"\n🔹 BIGVGAN CONFIGURATION (High Quality Preset):")
    print(f"   • Frequency Limit: UNCLAMPED (Full 0Hz - 11kHz)")
    print(f"   • Batch Size:      {TARGET_BATCH_SIZE} (Dynamic Setting)")
    print(f"   • Goal:            High Fidelity / No Ringing")
    
    # 3. Check Pretrained Vocoder
    pretrained = os.path.join("pretrained_checkpoints", "bigvgan_generator.pt")
    if os.path.exists(pretrained):
        print(f"\n✅ Found Base Vocoder: {pretrained}")
    else:
        # Fallback check
        alt = os.path.join("pretrained_checkpoints", "bigvgan_v2_22khz_80band_256x.pt")
        if os.path.exists(alt):
            print(f"\n✅ Found Base Vocoder: {alt}")
        else:
            print(f"\n❌ CRITICAL WARNING: Base Vocoder not found in 'pretrained_checkpoints'!")
    
    print(f"{'-'*60}\n")

# --- 🧹 CLEANUP BOT 🧹 ---
def cleanup_loop(checkpoint_dir, keep_n=3):
    """
    Runs in the background. 
    Keeps exactly the 'keep_n' newest checkpoints (e.g., last 3 epochs).
    """
    print(f"   🤖 Cleanup Bot activated (Keeping last {keep_n} checkpoints)...")
    while True:
        try:
            # 1. Find all Generator checkpoints (g_*)
            pattern = os.path.join(checkpoint_dir, "g_*")
            files = glob.glob(pattern)
            files = [f for f in files if os.path.isfile(f)]
            
            # If we have more than the limit
            if len(files) > keep_n:
                # 2. Sort by modification time (Oldest first)
                files.sort(key=os.path.getmtime)
                
                # 3. Delete everything except the last 'keep_n'
                to_delete = files[:-keep_n]
                
                for f in to_delete:
                    try:
                        print(f"   🗑️  Auto-Cleaning old epoch: {os.path.basename(f)}")
                        os.remove(f)
                        
                        # Also delete the matching Discriminator file (do_*)
                        do_file = f.replace(os.path.sep + "g_", os.path.sep + "do_")
                        if os.path.exists(do_file):
                            os.remove(do_file)
                            
                    except OSError:
                        pass
                        
        except Exception as e:
            print(f"   ⚠️ Cleanup Bot Error: {e}")
            
        # Check every 30 seconds
        time.sleep(30)

# --- SETUP FUNCTIONS ---
def fix_filelists(wav_dir, train_list_path, val_list_path):
    print("   🔧 Fixing filelists...")
    if not os.path.exists(wav_dir):
        print(f"❌ Error: Wav directory not found at {wav_dir}")
        sys.exit(1)

    files = [
        os.path.abspath(os.path.join(wav_dir, f.replace(".wav", ""))) 
        for f in os.listdir(wav_dir) 
        if f.endswith(".wav")
    ]
    
    if not files:
        print("❌ Error: No wav files found.")
        sys.exit(1)

    random.shuffle(files)
    split_idx = int(len(files) * 0.95)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    with open(train_list_path, "w") as f: f.write("\n".join(train_files))
    with open(val_list_path, "w") as f: f.write("\n".join(val_files))
    
    return len(files)

def force_write_golden_config(config_path, steps_per_epoch):
    """
    Writes the LONG HAUL config.
    Resets decay to 0.99995 to reach 100k steps without dying.
    UPDATED: Forces High Quality (fmax=None) and uses TARGET_BATCH_SIZE.
    """
    print(f"   ☢️  Enforcing High Quality Config (Batch {TARGET_BATCH_SIZE} | Unclamped Freqs)...")
    
    pretrained_ckpt = os.path.abspath(os.path.join("pretrained_checkpoints", "bigvgan_generator.pt"))
    if not os.path.exists(pretrained_ckpt):
        pretrained_ckpt = os.path.abspath(os.path.join("pretrained_checkpoints", "bigvgan_v2_22khz_80band_256x.pt"))
    
    golden_config = {
        "seed": 1234,
        "dist_config": {
            "dist_backend": "nccl",
            "dist_url": "tcp://localhost:54321",
            "world_size": 1
        },
        "sampling_rate": 22050,
        "n_mel_channels": 80,
        "num_mels": 80,
        "n_fft": 1024,
        "filter_length": 1024,
        "hop_size": 256,
        "hop_length": 256,
        "win_size": 1024,
        "win_length": 1024,
        
        # --- HIGH QUALITY SETTINGS (CRITICAL) ---
        "fmin": 40,            # <--- OPTIMIZED: Removes sub-bass mud
        "fmax": None,          # <--- None = No Limit (High Quality)
        "mel_fmin": 40,        # <--- OPTIMIZED: Focuses bands on voice
        "mel_fmax": None,      # <--- None = No Limit
        "fmax_for_loss": None, # <--- None = No Limit
        
        "activation": "snakebeta",
        "snake_log_interval": 10,
        "snake_logscale": True,
        "use_snake_at_generator": True, 
        "resblock": "1",
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [8, 8, 2, 2],
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [16, 16, 4, 4],
        "upsample_input_conv_kernel_size": 7,
        "upsample_input_conv_stride": 1, 
        "n_layers_q": 3,
        "use_spectral_norm": False,
        "use_weight_norm": True,
        "mpd_reshapes": [2, 3, 5, 7, 11], 
        "use_snake_at_discriminator": True,
        "discriminator_channel_mult": 1,
        "resolutions": [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]],
        
        "num_gpus": 0,
        "batch_size": TARGET_BATCH_SIZE, # <--- DYNAMIC
        
        # --- LONG HAUL SETTINGS ---
        "learning_rate": 0.0001,
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": 0.99995,  # Slow decay to survive to 100k
        "training_epochs": 100000,
        
        "stdout_interval": 5,
        "checkpoint_interval": steps_per_epoch, 
        "summary_interval": 100,
        "validation_interval": steps_per_epoch, 
        "checkpoint_path": "vocoder_checkpoints",
        "segment_size": 8192,
        "num_workers": 2, 
        "finetune_from_model": pretrained_ckpt,
        "training_files": "dataset/filelist_train.txt",
        "validation_files": "dataset/filelist_val.txt",
    }
    
    golden_config["model_config"] = golden_config.copy()
    
    with open(config_path, 'w') as f:
        json.dump(golden_config, f, indent=4)

def main():
    print_status_report()
    print(f"--- 🧠 Starting BigVGAN Fine-Tuning (High Quality Mode) ---")
    
    vocoder_dir = "BigVGAN"
    config_path = os.path.join(vocoder_dir, "configs", "finetune_22khz.json")
    wav_dir = os.path.join("dataset", "wavs")
    train_list = os.path.join("dataset", "filelist_train.txt")
    val_list = os.path.join("dataset", "filelist_val.txt")
    
    if not os.path.exists(vocoder_dir):
        print("❌ Error: 'BigVGAN' folder missing.")
        sys.exit(1)

    # 1. Prepare files and calculate steps
    total_files = fix_filelists(wav_dir, train_list, val_list)
    
    # Calculate Steps per Epoch (Dynamic based on TARGET_BATCH_SIZE)
    steps_per_epoch = math.ceil(total_files / TARGET_BATCH_SIZE)
    print(f"   📊 Dataset Size: {total_files} files")
    print(f"   ⏱️  1 Epoch = {steps_per_epoch} Steps (Will save every epoch)")

    # 2. Update config (Force High Quality settings)
    force_write_golden_config(config_path, steps_per_epoch)

    # 3. Start Cleanup Bot
    t = threading.Thread(target=cleanup_loop, args=("vocoder_checkpoints", 3), daemon=True)
    t.start()

    # 4. Launch Training
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(vocoder_dir) + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [
        sys.executable, 
        os.path.join(vocoder_dir, "train.py"),
        "--config", config_path,
        "--input_training_file", train_list,
        "--input_validation_file", val_list,
        "--list_input_unseen_validation_file", val_list,
        "--list_input_unseen_wavs_dir", wav_dir,
        "--checkpoint_path", "vocoder_checkpoints",
        "--checkpoint_interval", str(steps_per_epoch)
    ]
    
    print("   🚀 Launching engine...")
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n\n⏸️  Training Paused.")
    except Exception as e:
        print(f"\n❌ Error launching training: {e}")

if __name__ == "__main__":
    main()
