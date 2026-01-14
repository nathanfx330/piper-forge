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

# --- ⚙️  HARDWARE SETTINGS ---
# 4 = Safe (100% stable)
# 6 = Aggressive (Good for 10GB-12GB VRAM)
# 8 = High (Requires 12GB+ free or 16GB VRAM)
TARGET_BATCH_SIZE = 6

# --- 🕹️  INTERACTIVE MENU ---
def ask_training_mode():
    print(f"\n{'-'*60}")
    print(f"🎛️  BIGVGAN TRAINING COMMAND CENTER")
    print(f"{'-'*60}")
    print("Select your training strategy:")
    print(f"\n   [1] 🚀 STANDARD FINE-TUNE (Learning Rate: 0.0001)")
    print("       • Use for the first 0 - 50,000 steps.")
    print("       • Learns the voice shape quickly.")
    
    print(f"\n   [2] 💎 PRECISION POLISH   (Learning Rate: 0.00005)")
    print("       • Use if you hear metallic artifacts or flanging.")
    print("       • Half-speed training to refine texture.")
    
    while True:
        choice = input("\n👉 Select Option [1 or 2]: ").strip()
        if choice == "1":
            return 0.0001, 0.99995, "STANDARD"
        elif choice == "2":
            return 0.00005, 0.99999, "PRECISION"
        else:
            print("❌ Invalid input. Type 1 or 2.")

# --- 🔍 SYSTEM CHECK & FEEDBACK ---
def print_status_report(mode_name, lr, decay):
    print(f"\n{'-'*60}")
    print(f"📊 SYSTEM STATUS REPORT")
    print(f"{'-'*60}")
    
    print(f"🔹 PIPER CONTEXT:")
    print(f"   • Voice Name:    {VOICE_NAME}")
    print(f"   • Dataset:       dataset/wavs/")
    
    print(f"\n🔹 BIGVGAN STRATEGY: {mode_name}")
    print(f"   • Batch Size:    {TARGET_BATCH_SIZE}")
    print(f"   • Freq Range:    40Hz - UNLIMITED (High Quality)")
    print(f"   • Learning Rate: {lr}")
    print(f"   • Decay Rate:    {decay}")
    
    # Check Pretrained Vocoder
    pretrained = os.path.join("pretrained_checkpoints", "bigvgan_generator.pt")
    if os.path.exists(pretrained):
        print(f"\n✅ Found Base Vocoder: {pretrained}")
    else:
        alt = os.path.join("pretrained_checkpoints", "bigvgan_v2_22khz_80band_256x.pt")
        if os.path.exists(alt):
            print(f"\n✅ Found Base Vocoder: {alt}")
        else:
            print(f"\n❌ CRITICAL WARNING: Base Vocoder not found in 'pretrained_checkpoints'!")
    
    print(f"{'-'*60}\n")

# --- 🧹 CLEANUP BOT 🧹 ---
def cleanup_loop(checkpoint_dir, keep_n=3):
    print(f"   🤖 Cleanup Bot activated (Keeping last {keep_n} checkpoints)...")
    while True:
        try:
            pattern = os.path.join(checkpoint_dir, "g_*")
            files = glob.glob(pattern)
            files = [f for f in files if os.path.isfile(f)]
            
            if len(files) > keep_n:
                files.sort(key=os.path.getmtime)
                to_delete = files[:-keep_n]
                for f in to_delete:
                    try:
                        # Also delete discriminator
                        do_file = f.replace(os.path.sep + "g_", os.path.sep + "do_")
                        if os.path.exists(do_file): os.remove(do_file)
                        os.remove(f)
                        print(f"   🗑️  Auto-Cleaned: {os.path.basename(f)}")
                    except OSError: pass
        except Exception: pass
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

def force_write_golden_config(config_path, steps_per_epoch, lr, decay):
    print(f"   ☢️  Writing Config (LR={lr} | Fmin=40Hz)...")
    
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
        
        # --- AUDIO QUALITY SETTINGS ---
        "fmin": 40,            # 40Hz (Cleaner bass)
        "fmax": None,          # Unlimited Highs
        "mel_fmin": 40,        
        "mel_fmax": None,      
        "fmax_for_loss": None, 
        
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
        "batch_size": TARGET_BATCH_SIZE,
        
        # --- DYNAMIC TRAINING SETTINGS ---
        "learning_rate": lr,     # <--- User Selected
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": decay,       # <--- User Selected
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
    # 1. Ask User for Strategy
    lr, decay, mode_name = ask_training_mode()
    
    print_status_report(mode_name, lr, decay)
    
    vocoder_dir = "BigVGAN"
    config_path = os.path.join(vocoder_dir, "configs", "finetune_22khz.json")
    wav_dir = os.path.join("dataset", "wavs")
    train_list = os.path.join("dataset", "filelist_train.txt")
    val_list = os.path.join("dataset", "filelist_val.txt")
    
    if not os.path.exists(vocoder_dir):
        print("❌ Error: 'BigVGAN' folder missing.")
        sys.exit(1)

    # 2. Prepare files and calculate steps
    total_files = fix_filelists(wav_dir, train_list, val_list)
    steps_per_epoch = math.ceil(total_files / TARGET_BATCH_SIZE)
    print(f"   📊 Dataset Size: {total_files} files")
    print(f"   ⏱️  1 Epoch = {steps_per_epoch} Steps")

    # 3. Update config with USER SELECTED strategy
    force_write_golden_config(config_path, steps_per_epoch, lr, decay)

    # 4. Start Cleanup Bot
    t = threading.Thread(target=cleanup_loop, args=("vocoder_checkpoints", 3), daemon=True)
    t.start()

    # 5. Launch Training
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
