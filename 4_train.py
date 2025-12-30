import os
import sys
import glob
import subprocess
from config import *

def get_resume_checkpoint():
    """Decides whether to start fresh or resume."""
    # 1. Check for existing training checkpoints
    # They live deep in lightning_logs/version_x/checkpoints/
    search_pattern = os.path.join(TRAINING_DIR, "lightning_logs", "**", "*.ckpt")
    checkpoints = glob.glob(search_pattern, recursive=True)
    
    if checkpoints:
        # Find the newest one
        latest = max(checkpoints, key=os.path.getmtime)
        print(f"🔄 Resuming from existing checkpoint:\n   {os.path.basename(latest)}")
        return latest
    
    # 2. If none, use the Base Model
    if os.path.exists(BASE_MODEL_FILENAME):
        print(f"🆕 Starting fresh from Base Model:\n   {BASE_MODEL_FILENAME}")
        return BASE_MODEL_FILENAME
    
    print(f"❌ Error: No base model found at {BASE_MODEL_FILENAME}")
    sys.exit(1)

def main():
    print(f"--- 🚂 Starting Training: {VOICE_NAME} ---")
    print(f"    Quality: {QUALITY}")
    print(f"    Batch Size: {BATCH_SIZE}")
    print(f"    Max Epochs: {MAX_EPOCHS}")
    
    # 1. Setup Environment
    piper_src = os.path.join(PIPER_DIR, "src", "python")
    env = os.environ.copy()
    env["PYTHONPATH"] = piper_src + os.pathsep + env.get("PYTHONPATH", "")
    # Hide the massive amount of pytorch warnings
    env["PYTHONWARNINGS"] = "ignore"

    # 2. Get Checkpoint
    resume_ckpt = get_resume_checkpoint()

    # 3. Build Command
    # Note: We use --log_every_n_steps (underscores) based on your version
    cmd = [
        sys.executable, "-m", "piper_train",
        "--dataset-dir", TRAINING_DIR,
        "--accelerator", "gpu",
        "--devices", "1",
        "--batch-size", str(BATCH_SIZE),
        "--quality", QUALITY,
        "--resume_from_checkpoint", resume_ckpt,
        "--checkpoint-epochs", str(SAVE_EVERY_EPOCHS),
        "--precision", "32",
        "--max_epochs", str(MAX_EPOCHS),
        "--log_every_n_steps", "1" 
    ]

    # 4. Run
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n\n⏸️  Training Paused (Safe to exit).")
        print("   Run this script again to resume.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training Crashed: {e}")

if __name__ == "__main__":
    main()