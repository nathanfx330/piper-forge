import os
import shutil
import sys
import glob
from config import *

def remove_folder(path):
    if os.path.exists(path):
        try:
            print(f"💥 Destroying {path}...")
            shutil.rmtree(path)
        except OSError as e:
            print(f"❌ Error deleting {path}: {e}")

def remove_file(path):
    if os.path.exists(path):
        try:
            print(f"💥 Deleting file {path}...")
            os.remove(path)
        except OSError as e:
            print(f"❌ Error deleting {path}: {e}")

def main():
    print(f"--- ☢️  TOTAL NUCLEAR RESET: '{VOICE_NAME}' ---")
    print(f"    Current Config Quality: {QUALITY}")
    
    # LIST OF TARGETS TO DESTROY
    targets = []
    
    # 1. The Piper Brain
    if os.path.exists(TRAINING_DIR): targets.append(TRAINING_DIR)
    if os.path.exists("lightning_logs"): targets.append("lightning_logs")
    
    # 2. The Dataset (Slices & Metadata)
    if os.path.exists(DATASET_DIR): targets.append(DATASET_DIR)
    
    # 3. The Vocoder (BigVGAN) - EVERYTHING MUST GO
    if os.path.exists("vocoder_checkpoints"): targets.append("vocoder_checkpoints")
    if os.path.exists("vocoder_previews"): targets.append("vocoder_previews")
    
    # 4. The Old Vocoder Config (So it regenerates fresh)
    vocoder_config = os.path.join("BigVGAN", "configs", "finetune_22khz.json")

    # CHECK IF EMPTY
    if not targets and not os.path.exists(vocoder_config):
        print("\nℹ️  Clean slate detected. Nothing to kill.")
        return

    print("\n⚠️  WARNING: SCORCHED EARTH PROTOCOL")
    print("   The following folders will be PERMANENTLY DELETED:")
    for t in targets:
        print(f"   ❌ {t}/")
    if os.path.exists(vocoder_config):
        print(f"   ❌ {vocoder_config}")

    print("\n   🛡️  SAFE: 'raw_audio/'")
    print(f"   🛡️  SAFE: '{BASE_MODEL_FILENAME}'")

    confirm = input("\nType 'NUKE' to confirm total deletion: ").strip()
    
    if confirm == "NUKE":
        # Execute Order 66
        for t in targets:
            remove_folder(t)
            
        if os.path.exists(vocoder_config):
            remove_file(vocoder_config)

        print("\n✅ Total Wipe Complete.")
        print("-" * 40)
        print("👉 FRESH START CHECKLIST:")
        print("1. Run 'python 2_slice_and_transcribe.py'")
        print("2. Run 'python 3_preprocess.py'")
        print("3. Run 'python 4_train.py'")
        print("-" * 40)
    else:
        print("🚫 Cancelled.")

if __name__ == "__main__":
    main()
