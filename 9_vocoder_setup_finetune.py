import os
import sys
import subprocess
import json
import random
import shutil
from config import *

# --- CONFIGURATION ---
BIGVGAN_REPO = "https://github.com/NVIDIA/BigVGAN.git"
VOCODER_DIR = "BigVGAN"
CHECKPOINT_DIR = "pretrained_checkpoints"

# The "Universal Physics" Model (NVIDIA BigVGAN v2 22kHz)
HF_REPO_ID = "nvidia/BigVGAN_v2_22khz_80band_256x"
# Filename fixed to match NVIDIA repo structure
PRETRAINED_FILENAME = "bigvgan_generator.pt"

CONFIG_OUTPUT = os.path.join(VOCODER_DIR, "configs", "finetune_22khz.json")

def check_reset_protocol():
    """
    Asks the user if they want to wipe previous training progress
    to start fresh from the NVIDIA base model.
    """
    ckpt_dir = "vocoder_checkpoints"
    prev_dir = "vocoder_previews"
    
    # Check if there is anything to delete
    has_data = (os.path.exists(ckpt_dir) and os.listdir(ckpt_dir))
    
    if has_data:
        print(f"\n{'-'*50}")
        print(f"⚠️  PREVIOUS TRAINING DETECTED")
        print(f"{'-'*50}")
        print(f"   I found existing checkpoints in '{ckpt_dir}'.")
        print("   If you want to restart training from the NVIDIA Base,")
        print("   you must delete these files.\n")
        
        print("   [1] KEEP progress (Resume training)")
        print("   [2] RESET progress (Delete and start fresh)")
        
        choice = input("\n   Select option [1/2]: ").strip()
        
        if choice == "2":
            confirm = input("   💥 Are you sure? Type 'yes' to delete: ").strip().lower()
            if confirm == "yes":
                print("\n   🗑️  Cleaning up...")
                try:
                    shutil.rmtree(ckpt_dir)
                    print(f"      - Deleted {ckpt_dir}")
                    if os.path.exists(prev_dir):
                        shutil.rmtree(prev_dir)
                        print(f"      - Deleted {prev_dir}")
                    print("   ✅ Reset Complete. Ready for fresh training.")
                except Exception as e:
                    print(f"   ❌ Error deleting folders: {e}")
                    print("      (You might need to close other scripts/folders first)")
            else:
                print("   🚫 Reset cancelled.")
        else:
            print("   ℹ️  Progress preserved.")

def install_dependencies():
    print(f"\n--- 🛠️  Setting up BigVGAN Environment ---")
    
    if not os.path.exists(VOCODER_DIR):
        print(f"   📥 Cloning BigVGAN from {BIGVGAN_REPO}...")
        try:
            subprocess.run(["git", "clone", BIGVGAN_REPO], check=True)
        except FileNotFoundError:
            print("❌ Error: 'git' command not found. Please install Git.")
            sys.exit(1)
    else:
        print(f"   ✅ BigVGAN folder exists.")

    print("   📦 Checking/Installing dependencies...")
    reqs = ["alias-free-torch", "pesq", "auraloss", "vector-quantize-pytorch", "huggingface_hub"]
    for req in reqs:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", req], check=True, stdout=subprocess.DEVNULL)
            print(f"      - {req}: OK")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  Warning: Automatic install failed for '{req}'.")
    
    print("   (Note: Ignore 'pip dependency resolver' errors. BigVGAN requires newer Torch than Piper.)")

def get_pretrained_model():
    print(f"\n--- 🧠 Fetching Pretrained Foundation Model ---")
    from huggingface_hub import hf_hub_download
    
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    target_path = os.path.join(CHECKPOINT_DIR, PRETRAINED_FILENAME)
    
    if os.path.exists(target_path):
        print(f"   ✅ Found existing checkpoint: {target_path}")
        return target_path
    
    print(f"   ⬇️  Downloading NVIDIA BigVGAN v2 (22kHz)...")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID, 
            filename=PRETRAINED_FILENAME, 
            local_dir=CHECKPOINT_DIR
        )
        print("   ✅ Download complete.")
        return downloaded
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        sys.exit(1)

def prepare_filelists():
    print(f"\n--- 📄 Preparing Dataset Filelists ---")
    wavs_dir = os.path.join(DATASET_DIR, "wavs")
    
    if not os.path.exists(wavs_dir):
        print(f"❌ Error: {wavs_dir} missing. Run script 2 (slicing) first.")
        sys.exit(1)

    wav_files = [os.path.abspath(os.path.join(wavs_dir, f)) for f in os.listdir(wavs_dir) if f.endswith(".wav")]
    
    if not wav_files:
        print("❌ No WAV files found.")
        sys.exit(1)

    random.shuffle(wav_files)

    split_idx = int(len(wav_files) * 0.95)
    train_files = wav_files[:split_idx]
    val_files = wav_files[split_idx:]
    
    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    with open("dataset/filelist_train.txt", "w") as f:
        f.write("\n".join(train_files))
    with open("dataset/filelist_val.txt", "w") as f:
        f.write("\n".join(val_files))
        
    print(f"   ✅ Train Samples: {len(train_files)}")
    print(f"   ✅ Val Samples:   {len(val_files)}")

def create_finetune_config(pretrained_path):
    print(f"\n--- ⚙️  Generating Fine-Tuning Config ---")
    
    # 1. Training Parameters
    train_conf = {
        "batch_size": 4, 
        "learning_rate": 0.0001,
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": 0.999,
        "training_epochs": 100000,
        "stdout_interval": 5,
        "checkpoint_interval": 1000, 
        "summary_interval": 100,
        "validation_interval": 1000,
        "checkpoint_path": "vocoder_checkpoints",
        "segment_size": 8192,
        "finetune_from_model": os.path.abspath(pretrained_path)
    }

    # 2. Data Parameters
    data_conf = {
        "training_files": "dataset/filelist_train.txt",
        "validation_files": "dataset/filelist_val.txt",
        "sampling_rate": 22050,
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0
    }
    
    # 3. Model Parameters (FLATTENED - NO NESTING)
    # These must be at the root level for BigVGAN V2 to read them
    model_conf = {
        "upsample_initial_channel": 512,
        "resblock": "1",
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [8, 8, 2, 2],
        "upsample_kernel_sizes": [16, 16, 4, 4],
        "upsample_input_conv_kernel_size": 7,
        "upsample_input_conv_stride": 1, 
        "n_layers_q": 3,
        "use_spectral_norm": False,
        "use_weight_norm": True
    }

    # 4. Merge Everything into ONE flat dictionary
    config = {
        "seed": 1234,
        "dist_config": {
            "dist_backend": "nccl",
            "dist_url": "tcp://localhost:54321",
            "world_size": 1
        },
        **train_conf, 
        **data_conf,
        **model_conf  # Merging model params to root
    }

    if not os.path.exists(os.path.dirname(CONFIG_OUTPUT)):
        os.makedirs(os.path.dirname(CONFIG_OUTPUT))

    with open(CONFIG_OUTPUT, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"   ✅ Config saved to: {CONFIG_OUTPUT}")

def main():
    check_reset_protocol()
    install_dependencies()
    pretrained_path = get_pretrained_model()
    prepare_filelists()
    create_finetune_config(pretrained_path)
    
    print("\n--- 🎉 Vocoder Setup Complete ---")
    print("Action: Run 'python 10_train_vocoder.py' to begin fine-tuning.")

if __name__ == "__main__":
    main()
