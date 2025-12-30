import os
import sys
import glob
import shutil
import subprocess
import json
from config import *

def main():
    print(f"--- 📦 Exporting Final Model: {VOICE_NAME} ---")

    # 1. Find the Best Checkpoint
    # Piper saves the best model as 'epoch=xxxx.ckpt' and the latest as 'last.ckpt'
    # We prefer the numbered epoch if available, as 'last' might be interrupted.
    search_pattern = os.path.join(TRAINING_DIR, "lightning_logs", "**", "*.ckpt")
    checkpoints = glob.glob(search_pattern, recursive=True)
    
    if not checkpoints:
        print(f"❌ Error: No checkpoints found in {TRAINING_DIR}")
        return

    # Filter out 'last.ckpt' to find the best calculated epoch, 
    # unless 'last.ckpt' is the only thing there.
    numbered_ckpts = [c for c in checkpoints if "epoch=" in c]
    
    if numbered_ckpts:
        best_ckpt = max(numbered_ckpts, key=os.path.getmtime)
    else:
        best_ckpt = max(checkpoints, key=os.path.getmtime)

    print(f"   Selected Brain: {os.path.basename(best_ckpt)}")

    # 2. Prepare Output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    final_onnx = os.path.join(OUTPUT_DIR, f"{VOICE_NAME}.onnx")
    final_conf = os.path.join(OUTPUT_DIR, f"{VOICE_NAME}.onnx.json")

    # 3. Run Export
    print("   Converting to ONNX (Optimizing)...")
    
    piper_src = os.path.join(PIPER_DIR, "src", "python")
    env = os.environ.copy()
    env["PYTHONPATH"] = piper_src + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [
        sys.executable, "-m", "piper_train.export_onnx",
        best_ckpt,
        final_onnx
    ]

    try:
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Export Failed: {e}")
        return

    # 4. Handle Config
    # We load the training config and save a clean version next to the model
    src_config = os.path.join(TRAINING_DIR, "config.json")
    if os.path.exists(src_config):
        shutil.copy(src_config, final_conf)
        print("   Config bundled.")
    else:
        print("⚠️  Warning: Could not find config.json to bundle.")

    # 5. Success Message
    print("\n--- 🎉 EXPORT SUCCESSFUL ---")
    print(f"Files saved to: '{OUTPUT_DIR}/'")
    print(f"1. {os.path.basename(final_onnx)}")
    print(f"2. {os.path.basename(final_conf)}")
    
    print("\n👇 Run this command to verify:")
    piper_exe = os.path.join(PIPER_DIR, "piper")
    print(f"echo 'This is the final version of the model.' | {piper_exe} --model {final_onnx} --output_file final_test.wav")

if __name__ == "__main__":
    main()