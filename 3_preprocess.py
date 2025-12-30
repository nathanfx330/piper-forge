import os
import sys
import subprocess
from config import *

def main():
    print(f"--- ⚙️  Preprocessing Data for '{VOICE_NAME}' ---")

    # 1. Setup Python Path
    # We need to tell Python where the Piper code lives
    piper_src = os.path.join(PIPER_DIR, "src", "python")
    if not os.path.exists(piper_src):
        print(f"❌ Error: Could not find piper source at {piper_src}")
        return

    # Add piper to the environment variables for this command
    env = os.environ.copy()
    env["PYTHONPATH"] = piper_src + os.pathsep + env.get("PYTHONPATH", "")

    print(f"   Input:  {DATASET_DIR}/")
    print(f"   Output: {TRAINING_DIR}/")
    print(f"   Specs:  {LANGUAGE_CODE} | {SAMPLE_RATE}Hz")

    # 2. Build Command
    # python -m piper_train.preprocess ...
    cmd = [
        sys.executable, "-m", "piper_train.preprocess",
        "--language", LANGUAGE_CODE,
        "--input-dir", DATASET_DIR,
        "--output-dir", TRAINING_DIR,
        "--dataset-format", "ljspeech",
        "--single-speaker",
        "--sample-rate", str(SAMPLE_RATE)
    ]

    # 3. Run
    try:
        subprocess.run(cmd, check=True, env=env)
        
        print("\n--- ✅ Preprocessing Complete ---")
        print(f"Folder '{TRAINING_DIR}/' is now populated with:")
        print("  - config.json")
        print("  - dataset.jsonl")
        print("  - tensors (.pt files)")
        print("\nNext Step: Run Script 4 to START TRAINING.")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Preprocessing Failed: {e}")

if __name__ == "__main__":
    main()