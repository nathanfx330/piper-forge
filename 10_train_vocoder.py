import os
import sys
import subprocess
import json
import random
from config import *

def fix_filelists(wav_dir, train_list_path, val_list_path):
    """
    The BigVGAN V2 code automatically appends '.wav' to files in the list.
    We must generate a filelist WITHOUT extensions to prevent 'file.wav.wav' errors.
    """
    print("   🔧 Fixing filelists (Stripping extensions)...")
    
    if not os.path.exists(wav_dir):
        print(f"❌ Error: Wav directory not found at {wav_dir}")
        sys.exit(1)

    # Get all .wav files but STRIP the extension for the text file
    # /path/to/file.wav -> /path/to/file
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
    
    with open(train_list_path, "w") as f:
        f.write("\n".join(train_files))
        
    with open(val_list_path, "w") as f:
        f.write("\n".join(val_files))
        
    print(f"      - Fixed {len(train_files)} training paths.")
    print(f"      - Fixed {len(val_files)} validation paths.")

def force_write_golden_config(config_path, vocoder_dir):
    print(f"   ☢️  Overwriting config with Golden V2 Template...")
    
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
        "fmin": 0,
        "fmax": 8000,
        "mel_fmin": 0,
        "mel_fmax": 8000,
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
        "num_workers": 2, 
        "finetune_from_model": pretrained_ckpt,
        "training_files": "dataset/filelist_train.txt",
        "validation_files": "dataset/filelist_val.txt",
    }
    
    golden_config["model_config"] = golden_config.copy()
    
    with open(config_path, 'w') as f:
        json.dump(golden_config, f, indent=4)

def main():
    print(f"--- 🧠 Starting BigVGAN Fine-Tuning for '{VOICE_NAME}' ---")
    
    vocoder_dir = "BigVGAN"
    config_path = os.path.join(vocoder_dir, "configs", "finetune_22khz.json")
    wav_dir = os.path.join("dataset", "wavs")
    train_list = os.path.join("dataset", "filelist_train.txt")
    val_list = os.path.join("dataset", "filelist_val.txt")
    
    if not os.path.exists(vocoder_dir):
        print("❌ Error: 'BigVGAN' folder missing.")
        sys.exit(1)

    # 1. FIX THE FILE LISTS (Remove .wav extension)
    fix_filelists(wav_dir, train_list, val_list)

    # 2. FIX THE CONFIG
    force_write_golden_config(config_path, vocoder_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(vocoder_dir) + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [
        sys.executable, 
        os.path.join(vocoder_dir, "train.py"),
        "--config", config_path,
        "--input_training_file", train_list,
        "--input_validation_file", val_list,
        # Override unseen validation to prevent crash
        "--list_input_unseen_validation_file", val_list,
        "--list_input_unseen_wavs_dir", wav_dir,
        "--checkpoint_path", "vocoder_checkpoints"
    ]
    
    print(f"   Config: {os.path.basename(config_path)}")
    print("   🚀 Launching engine...\n")
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n\n⏸️  Training Paused.")
    except Exception as e:
        print(f"\n❌ Error launching training: {e}")

if __name__ == "__main__":
    main()
