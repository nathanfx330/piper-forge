import os
import sys
import subprocess
import re
import time
import glob
import json
from config import *

# --- SETTINGS ---
OUTPUT_WAVS_DIR = "generated_wavs"
TXT_INPUT_DIR = "txts"
SETTINGS_FILE = "tts_settings.json"
MODEL_PATH = os.path.join(OUTPUT_DIR, f"{VOICE_NAME}.onnx")
PIPER_BINARY = os.path.join(PIPER_DIR, "piper")

# Fix for Windows exe extension
if sys.platform == "win32" and not PIPER_BINARY.endswith(".exe"):
    PIPER_BINARY += ".exe"

def ensure_setup():
    # 1. Check Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("   Did you run 6_export.py?")
        sys.exit(1)
    
    # 2. Check Output Dir
    if not os.path.exists(OUTPUT_WAVS_DIR):
        os.makedirs(OUTPUT_WAVS_DIR)
        print(f"📂 Created output folder: {OUTPUT_WAVS_DIR}/")

    # 3. Check Text Input Dir
    if not os.path.exists(TXT_INPUT_DIR):
        os.makedirs(TXT_INPUT_DIR)
        print(f"📂 Created text folder:   {TXT_INPUT_DIR}/")

def get_settings():
    """Reads (or creates) the JSON settings file."""
    path = os.path.join(TXT_INPUT_DIR, SETTINGS_FILE)
    
    defaults = {
        "length_scale": 1.0,      # Speed: < 1.0 is Faster, > 1.0 is Slower
        "noise_scale": 0.667,     # Variability/Emotion (0.0 to 1.0)
        "noise_w": 0.8,           # Phoneme Width Noise (Rhythm)
        "sentence_silence": 0.2   # Seconds of silence between sentences
    }
    
    # Create if missing
    if not os.path.exists(path):
        try:
            with open(path, 'w') as f:
                json.dump(defaults, f, indent=4)
            print(f"⚙️  Created default settings: {path}")
        except Exception as e:
            print(f"⚠️  Could not write settings: {e}")
            return defaults

    # Read
    try:
        with open(path, 'r') as f:
            user_settings = json.load(f)
            return {**defaults, **user_settings}
    except Exception as e:
        print(f"⚠️  Error reading {SETTINGS_FILE}: {e}")
        return defaults

def get_filename_from_text(text):
    """Creates a safe filename from the first 5 words."""
    clean_text = re.sub(r'[^\w\s]', '', text)
    words = clean_text.split()
    slug_words = words[:5]
    slug = "_".join(slug_words).lower()
    
    if not slug:
        slug = f"tts_{int(time.time())}"
        
    return f"{slug}.wav"

def run_piper(text, output_filename):
    """Core function to send text to Piper with custom settings"""
    output_path = os.path.join(OUTPUT_WAVS_DIR, output_filename)
    
    # Load settings fresh every time (Hot-Reload)
    settings = get_settings()
    
    try:
        cmd = [
            PIPER_BINARY,
            "--model", MODEL_PATH,
            "--output_file", output_path,
            "--length_scale", str(settings['length_scale']),
            "--noise_scale", str(settings['noise_scale']),
            "--noise_w", str(settings['noise_w']),
            "--sentence_silence", str(settings['sentence_silence'])
        ]
        
        process = subprocess.Popen(
            cmd, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        process.communicate(input=text.encode('utf-8'))
        
        if process.returncode == 0:
            return output_path
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def mode_interactive():
    print(f"\n--- ⌨️  Interactive Mode ---")
    print(f"💡 Tip: You can edit './{TXT_INPUT_DIR}/{SETTINGS_FILE}' while this is running!")
    print("Type 'back' to return to menu.")
    
    while True:
        try:
            text = input("\n📝 Enter text: ").strip()
            
            if not text: continue
            if text.lower() in ["back", "exit", "quit"]: break

            print("   Thinking...", end="\r")
            
            filename = get_filename_from_text(text)
            saved_file = run_piper(text, filename)
            
            if saved_file:
                s = get_settings()
                print(f"✅ Saved: {os.path.basename(saved_file)} (Spd: {s['length_scale']} | Noi: {s['noise_scale']})")
            else:
                print(f"❌ Failed to generate audio.")
                
        except KeyboardInterrupt:
            break

def mode_read_file():
    print(f"\n--- 📄 File Reader Mode ---")
    print(f"Scanning folder: ./{TXT_INPUT_DIR}/")
    
    all_files = sorted(glob.glob(os.path.join(TXT_INPUT_DIR, "*.txt")))
    
    if not all_files:
        print("⚠️  No .txt files found.")
        print(f"   Please put files into the '{TXT_INPUT_DIR}' folder first.")
        return

    print("\nAvailable Files:")
    for i, f in enumerate(all_files):
        print(f"   {i+1}. {os.path.basename(f)}")
    
    choice = input("\nSelect number (or 'q' to cancel): ").strip()
    if choice.lower() == 'q': return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(all_files):
            target_file = all_files[idx]
            
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                print("❌ File is empty.")
                return

            print(f"\n📖 Reading: {os.path.basename(target_file)}")
            s = get_settings()
            print(f"⚙️  Using: Speed={s['length_scale']}, Noise={s['noise_scale']}")
            print("   Generating audio...")
            
            out_name = os.path.basename(target_file).replace(".txt", ".wav")
            
            start_time = time.time()
            saved_file = run_piper(content, out_name)
            end_time = time.time()
            
            if saved_file:
                print(f"✅ Done in {end_time - start_time:.1f}s")
                print(f"💾 Saved to: {OUTPUT_WAVS_DIR}/{out_name}")
            else:
                print("❌ Generation failed.")
        else:
            print("❌ Invalid number.")
    except ValueError:
        print("❌ Invalid input.")

def main():
    ensure_setup()
    
    # --- FIX: Force creation of settings file immediately on startup ---
    get_settings() 
    # -----------------------------------------------------------------

    while True:
        print("\n" + "="*50)
        print(f"🎙️  Piper TTS: {VOICE_NAME}")
        print("="*50)
        print("1. ⌨️  Type Interactively")
        print("2. 📄 Read from ./txts folder")
        print("3. 🚪 Exit")
        
        choice = input("\nChoose option: ").strip()
        
        if choice == "1":
            mode_interactive()
        elif choice == "2":
            mode_read_file()
        elif choice == "3" or choice.lower() == "quit":
            print("👋 Bye")
            sys.exit(0)
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main()
