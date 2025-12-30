import os
import sys
from config import *

def create_folders():
    print("--- 🏗️  Building Directory Structure ---")
    folders = [RAW_AUDIO_DIR, DATASET_DIR, TRAINING_DIR, OUTPUT_DIR]
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✅ Created: {folder}/")
        else:
            print(f"ℹ️  Exists:  {folder}/")

def check_piper_code():
    print("\n--- 🔍 Checking for Piper Engine ---")
    piper_module_path = os.path.join(PIPER_DIR, "src", "python")
    
    if not os.path.exists(piper_module_path):
        print(f"❌ ERROR: Could not find Piper source code at: {piper_module_path}")
        print("   Action: Copy your existing 'piper' folder into this directory.")
        sys.exit(1)
    else:
        print(f"✅ Found Piper code at: {PIPER_DIR}/")

def check_base_model():
    print(f"\n--- 🧠 Checking Base Model ({QUALITY}) ---")
    
    if os.path.exists(BASE_MODEL_FILENAME):
        print(f"✅ Base model found: {BASE_MODEL_FILENAME}")
        return

    # MANUAL INSTRUCTION BLOCK
    print(f"❌ MISSING: The base model file is required to start training.")
    print("⚠️  You must download this manually.")
    print("-" * 60)
    print(f"1. Open this URL in your browser:")
    print(f"   {BASE_MODEL_URL}")
    print(f"\n   (If that link is dead, browse here: https://huggingface.co/rhasspy/piper-checkpoints/tree/main )")
    print(f"\n2. Download the .ckpt file (usually ~400MB - 800MB)")
    print(f"\n3. Move the file into this folder:")
    print(f"   {os.getcwd()}")
    print(f"\n4. RENAME the file to exactly:")
    print(f"   {BASE_MODEL_FILENAME}")
    print("-" * 60)
    
    sys.exit(1)

def main():
    print(f"Setup initiated for Voice: '{VOICE_NAME}'\n")
    
    create_folders()
    check_piper_code()
    check_base_model()
    
    print("\n--- 🎉 Setup Complete ---")
    print(f"1. Put your audio files (.wav or .mp3) inside: '{RAW_AUDIO_DIR}/'")
    print("2. Run 'python 2_slice_and_transcribe.py'")

if __name__ == "__main__":
    main()