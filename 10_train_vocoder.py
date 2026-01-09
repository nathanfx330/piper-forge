import os
import sys
import subprocess
from config import *

def main():
    print(f"--- 🧠 Starting BigVGAN Fine-Tuning for '{VOICE_NAME}' ---")
    
    vocoder_dir = "BigVGAN"
    # This points to the config we generated in script 9
    config_path = os.path.join(vocoder_dir, "configs", "finetune_22khz.json")
    
    # 1. Sanity Checks
    if not os.path.exists(vocoder_dir):
        print("❌ Error: 'BigVGAN' folder missing.")
        print("   Action: Run '9_vocoder_setup_finetune.py' first.")
        sys.exit(1)

    if not os.path.exists(config_path):
        print(f"❌ Error: Config file missing at {config_path}")
        print("   Action: Run '9_vocoder_setup_finetune.py' first.")
        sys.exit(1)

    # 2. Environment Setup
    # We need to add BigVGAN to the PYTHONPATH so its internal imports work
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(vocoder_dir) + os.pathsep + env.get("PYTHONPATH", "")
    
    # 3. Build Command
    # python BigVGAN/train.py --config BigVGAN/configs/finetune_22khz.json
    cmd = [
        sys.executable, 
        os.path.join(vocoder_dir, "train.py"),
        "--config", config_path
    ]
    
    print(f"   Config: {os.path.basename(config_path)}")
    print(f"   Output: vocoder_checkpoints/")
    print("-" * 50)
    print("⚠️  TRAINING ADVICE:")
    print("   1. Since we are Fine-Tuning a pretrained model, loss will start LOW.")
    print("   2. You do NOT need 100k steps. Check results after 5k, 10k, and 20k steps.")
    print("   3. Stop manually using Ctrl+C when satisfied.")
    print("-" * 50)
    print("   🚀 Launching engine...\n")
    
    # 4. Run
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n\n⏸️  Training Paused.")
        print("   Run this script again to resume exactly where you left off.")
    except Exception as e:
        print(f"\n❌ Error launching training: {e}")

if __name__ == "__main__":
    main()
