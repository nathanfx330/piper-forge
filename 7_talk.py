import os
import sys
import subprocess
import re  # Added for text cleaning
import time
from config import *

# --- SETTINGS ---
OUTPUT_WAVS_DIR = "generated_wavs"
MODEL_PATH = os.path.join(OUTPUT_DIR, f"{VOICE_NAME}.onnx")
PIPER_BINARY = os.path.join(PIPER_DIR, "piper")

# Fix for Windows exe extension
if sys.platform == "win32" and not PIPER_BINARY.endswith(".exe"):
    PIPER_BINARY += ".exe"

def ensure_setup():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("   Did you run 6_export.py?")
        sys.exit(1)
    
    if not os.path.exists(OUTPUT_WAVS_DIR):
        os.makedirs(OUTPUT_WAVS_DIR)
        print(f"📂 Created output folder: {OUTPUT_WAVS_DIR}/")

def get_filename_from_text(text):
    """Creates a safe filename from the first 5 words."""
    # 1. Remove anything that isn't a letter, number, or space
    clean_text = re.sub(r'[^\w\s]', '', text)
    
    # 2. Split into words
    words = clean_text.split()
    
    # 3. Take up to 5 words
    slug_words = words[:5]
    
    # 4. Join with underscores
    slug = "_".join(slug_words).lower()
    
    # 5. Fallback: If text was only symbols (e.g. "!!!")
    if not slug:
        slug = f"tts_{int(time.time())}"
        
    return f"{slug}.wav"

def generate_audio(text):
    filename = get_filename_from_text(text)
    output_path = os.path.join(OUTPUT_WAVS_DIR, filename)

    try:
        cmd = [
            PIPER_BINARY,
            "--model", MODEL_PATH,
            "--output_file", output_path
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

def main():
    ensure_setup()
    
    print("-" * 50)
    print(f"🎙️  Interactive Mode: {VOICE_NAME}")
    print(f"📂 Saving audio to: ./{OUTPUT_WAVS_DIR}/")
    print("❌ Type 'exit' or 'quit' to stop.")
    print("-" * 50)

    while True:
        try:
            text = input("\n📝 Enter text: ").strip()
            
            if not text:
                continue
            
            if text.lower() in ["exit", "quit"]:
                print("👋 Bye!")
                break

            print("   Thinking...", end="\r")
            saved_file = generate_audio(text)
            
            if saved_file:
                # We calculate the relative path just for cleaner printing
                rel_path = os.path.relpath(saved_file)
                print(f"✅ Saved: {rel_path}   ")
            else:
                print(f"❌ Failed to generate audio.")
                
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break

if __name__ == "__main__":
    main()