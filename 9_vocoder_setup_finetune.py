import os
import sys
import subprocess
import json
import random
from config import *

# --- CONFIGURATION ---
BIGVGAN_REPO = "https://github.com/NVIDIA/BigVGAN.git"
VOCODER_DIR = "BigVGAN"
CHECKPOINT_DIR = "pretrained_checkpoints"

# The "Universal Physics" Model (NVIDIA BigVGAN v2 22kHz)
# This model matches Piper's 22050Hz / 256 Hop / 80 Mel specs exactly.
HF_REPO_ID = "nvidia/BigVGAN_v2_22khz_80band_256x"

# CORRECTION: NVIDIA names the file 'bigvgan_generator.pt' inside the repo
PRETRAINED_FILENAME = "bigvgan_generator.pt"

CONFIG_OUTPUT = os.path.join(VOCODER_DIR, "configs", "finetune_22khz.json")

def install_dependencies():
    print(f"--- 🛠️  Setting up BigVGAN Environment ---")
    
    # 1. Clone Repo
    if not os.path.exists(VOCODER_DIR):
        print(f"   📥 Cloning BigVGAN from {BIGVGAN_REPO}...")
        try:
            subprocess.run(["git", "clone", BIGVGAN_REPO], check=True)
        except FileNotFoundError:
            print("❌ Error: 'git' command not found. Please install Git.")
            sys.exit(1)
    else:
        print(f"   ✅ BigVGAN folder exists.")

    # 2. Install Python Requirements
    print("   📦 Checking/Installing dependencies...")
    # These are specific to BigVGAN. 'alias-free-torch' is crucial for the activation functions.
    reqs = ["alias-free-torch", "pesq", "auraloss", "vector-quantize-pytorch", "huggingface_hub"]
    
    for req in reqs:
        try:
            # We use the current sys.executable to ensure it installs in the active environment
            subprocess.run([sys.executable, "-m", "pip", "install", req], check=True, stdout=subprocess.DEVNULL)
            print(f"      - {req}: OK")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  Warning: Automatic install failed for '{req}'. You might need to install it manually.")
    
    print("   (Note: Ignore 'pip dependency resolver' errors regarding torch versions. BigVGAN requires newer Torch than Piper.)")

def get_pretrained_model():
    print(f"\n--- 🧠 Fetching Pretrained Foundation Model ---")
    
    # Import inside function to ensure it's available after install_dependencies
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

    # BigVGAN reads pure wav paths. It computes Mels on the fly.
    wav_files = [os.path.abspath(os.path.join(wavs_dir, f)) for f in os.listdir(wavs_dir) if f.endswith(".wav")]
    
    if not wav_files:
        print("❌ No WAV files found.")
        sys.exit(1)

    # Shuffle to prevent bias in early training steps
    random.shuffle(wav_files)

    # 95% Train, 5% Val
    split_idx = int(len(wav_files) * 0.95)
    train_files = wav_files[:split_idx]
    val_files = wav_files[split_idx:]
    
    # Ensure dataset folder exists for the list files
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
    
    # This config matches the NVIDIA V2 architecture EXACTLY.
    # Changing model params (h) will cause the pretrained weights to fail loading.
    config = {
        "seed": 1234,
        "dist_config": {
            "dist_backend": "nccl",
            "dist_url": "tcp://localhost:54321",
            "world_size": 1
        },
        "model_config": {
            "h": {
                "upsample_initial_channel": 512,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [8, 8, 2, 2], # 8*8*2*2 = 256 Hop (Matches Piper 22k)
                "upsample_kernel_sizes": [16, 16, 4, 4],
                "upsample_input_conv_kernel_size": 7,
                "upsample_input_conv_stride": 1, 
                "n_layers_q": 3,
                "use_spectral_norm": False,
                "use_weight_norm": True
            }
        },
        "train_config": {
            "batch_size": 4,           # Low batch size for stability during fine-tuning
            "learning_rate": 0.0001,   # Low LR is critical for fine-tuning
            "adam_b1": 0.8,
            "adam_b2": 0.99,
            "lr_decay": 0.999,
            "training_epochs": 100000, # This effectively means "steps" in BigVGAN
            "stdout_interval": 5,
            "checkpoint_interval": 1000, 
            "summary_interval": 100,
            "validation_interval": 1000,
            "checkpoint_path": "vocoder_checkpoints",
            "segment_size": 8192,      # ~370ms audio segment per step
            
            # CRITICAL: This tells BigVGAN to load the NVIDIA weights
            "finetune_from_model": os.path.abspath(pretrained_path)
        },
        # We also need data config to tell it where to look
        "data_config": {
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
    }

    if not os.path.exists(os.path.dirname(CONFIG_OUTPUT)):
        os.makedirs(os.path.dirname(CONFIG_OUTPUT))

    with open(CONFIG_OUTPUT, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"   ✅ Config saved to: {CONFIG_OUTPUT}")

def main():
    install_dependencies()
    pretrained_path = get_pretrained_model()
    prepare_filelists()
    create_finetune_config(pretrained_path)
    
    print("\n--- 🎉 Vocoder Setup Complete ---")
    print("Action: Run 'python 10_train_vocoder.py' to begin fine-tuning.")

if __name__ == "__main__":
    main()
