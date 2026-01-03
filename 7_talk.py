import os
import sys
import subprocess
import re
import time
import glob
import json
import winsound
import wave
import struct

# --- CONFIGURATION ---
# Check config.py first, otherwise use defaults below
try:
    from config import *
except ImportError:
    # --- MANUAL CONFIG (If config.py is missing) ---
    PIPER_DIR = r"C:\Path\To\Piper_Folder" 
    VOICE_NAME = "en_US-libritts-high" 
    OUTPUT_DIR = "output"
    
    # UPDATE: Set this to your actual models folder name
    # Based on your input, it seems to be "final_models"
    MODELS_DIR = "final_models" 

# Ensure MODELS_DIR is defined if config.py didn't have it
if 'MODELS_DIR' not in locals():
    MODELS_DIR = "final_models"

# --- PATHS ---
OUTPUT_WAVS_DIR = os.path.join(OUTPUT_DIR, "generated_wavs")
TXT_INPUT_DIR = "txts"
TEMP_DIR = "temp_chunks"
SETTINGS_FILE = os.path.join(TXT_INPUT_DIR, "tts_settings.json")

# Default model path (fallback if no version found)
DEFAULT_MODEL_PATH = os.path.join(PIPER_DIR, f"{VOICE_NAME}.onnx")
PIPER_BINARY = os.path.join(PIPER_DIR, "piper.exe")

def ensure_setup():
    """Initializes folders and checks for required binaries."""
    # We check relative to current script location
    for d in [OUTPUT_DIR, OUTPUT_WAVS_DIR, TXT_INPUT_DIR, TEMP_DIR, MODELS_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

    if not os.path.exists(PIPER_BINARY):
        print(f"❌ Error: Piper executable not found at {PIPER_BINARY}")
        print(f"   Please check PIPER_DIR in the script.")
        sys.exit(1)

def _get_default_settings():
    """Returns the hardcoded default settings."""
    return {
        "autoplay": True,
        "default_version": "version_5", # Set your preference here for resets
        "length_scale": 1.0,
        "noise_scale": 0.667,
        "noise_w": 0.8,
        "sentence_silence": 0.2
    }

def get_settings():
    """Reads or creates the runtime JSON settings file."""
    defaults = _get_default_settings()
    
    if not os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(defaults, f, indent=4)
            print(f"Generated default settings file: {SETTINGS_FILE}")
        except Exception as e:
            print(f"Error creating settings file: {e}")
            return defaults

    try:
        with open(SETTINGS_FILE, 'r') as f:
            user_data = json.load(f)
            return {**defaults, **user_data} 
    except:
        return defaults

def regenerate_default_settings():
    """Overwrites the settings JSON file with default values."""
    defaults = _get_default_settings()
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(defaults, f, indent=4)
        print(f"\n✅ Default settings regenerated in '{SETTINGS_FILE}'")
        print(f"   Default version set to: '{defaults['default_version']}'")
    except Exception as e:
        print(f"\n❌ Error regenerating settings: {e}")
    input("Press Enter to continue...")

# --- MODEL RESOLUTION ---

def find_model_path(version_name=None):
    """
    Locates the .onnx file based on the folder name in MODELS_DIR.
    """
    if not version_name:
        return DEFAULT_MODEL_PATH

    # Look inside the configured models folder (e.g. final_models/version_5)
    target_folder = os.path.join(MODELS_DIR, version_name)
    target_file_in_root = os.path.join(MODELS_DIR, f"{version_name}.onnx")

    # 1. Check if it's a folder (Priority)
    if os.path.isdir(target_folder):
        # A. Check for specific name match inside folder (version_5.onnx)
        specific = os.path.join(target_folder, f"{version_name}.onnx")
        if os.path.exists(specific): return specific
        
        # B. Check for standard 'model.onnx' (Common in training exports)
        standard = os.path.join(target_folder, "model.onnx")
        if os.path.exists(standard): return standard
        
        # C. Grab the first .onnx found (Fallback)
        onnx_files = glob.glob(os.path.join(target_folder, "*.onnx"))
        if onnx_files: return onnx_files[0]

    # 2. Check if it's a direct file in final_models/
    if os.path.exists(target_file_in_root):
        return target_file_in_root

    print(f"⚠️  Model version '{version_name}' not found in '{MODELS_DIR}'. Using default fallback.")
    return DEFAULT_MODEL_PATH

# --- AUDIO UTILS ---

def create_silence(duration_ms, output_path, sample_rate=22050):
    num_samples = int((duration_ms / 1000.0) * sample_rate)
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setparams((1, 2, sample_rate, num_samples, 'NONE', 'not compressed'))
        wav_file.writeframes(struct.pack('<h', 0) * num_samples)

def merge_wavs(file_list, final_output):
    if not file_list: return
    data = []
    params = None
    
    valid_files = [f for f in file_list if os.path.exists(f)]
    if not valid_files: return

    try:
        for f in valid_files:
            with wave.open(f, 'rb') as w:
                if not params: params = w.getparams()
                data.append(w.readframes(w.getnframes()))
        
        with wave.open(final_output, 'wb') as out:
            out.setparams(params)
            for frames in data:
                out.writeframes(frames)
    except Exception as e:
        print(f"⚠️ Error merging: {e}")

# --- PROCESSING ---

def process_text_to_audio(input_text, final_filename):
    settings = get_settings()
    
    # --- MODEL SELECTION LOGIC ---
    # 1. Check for manual override in text: "version: v1"
    version_match = re.search(r'(?:^|\n)version:\s*([\w\-\.]+)', input_text, re.IGNORECASE)
    
    active_model_path = DEFAULT_MODEL_PATH
    clean_text = input_text

    if version_match:
        # CASE A: Tag found in text
        version_name = version_match.group(1).strip()
        print(f"   🔍 Version Override (Tag): {version_name}")
        active_model_path = find_model_path(version_name)
        # Remove tag from text
        clean_text = re.sub(r'(?:^|\n)version:\s*[\w\-\.]+', '', input_text, flags=re.IGNORECASE).strip()
    
    elif settings.get("default_version"):
        # CASE B: Setting found in JSON (e.g., "version_5")
        version_name = settings["default_version"].strip()
        # Only print if we are actually generating
        print(f"   🔍 Version Default (Settings): {version_name}")
        active_model_path = find_model_path(version_name)
        clean_text = input_text
    
    else:
        # CASE C: Fallback to global default
        print(f"   🤖 Using Global Default Model")
        active_model_path = DEFAULT_MODEL_PATH
        clean_text = input_text

    # 2. Tag Cleaning
    text = re.sub(r'</?speak>', '', clean_text).strip()
    text = re.sub(r'<down\s*/?>', '...<DOWN_TRIGGER>', text, flags=re.IGNORECASE)
    
    if not text:
        print("   ⚠️ Input text is empty. Skipping.")
        return None

    parts = re.split(r'(<break\s+time=["\']\d+[ms]*["\']\s*/>)', text)
    
    chunk_files = []
    current_sample_rate = 22050 

    for i, part in enumerate(parts):
        part = part.strip()
        if not part: continue
        
        chunk_path = os.path.join(TEMP_DIR, f"chunk_{i}.wav")
        break_match = re.search(r'time=["\'](\d+)(ms|s)["\']', part)
        
        if break_match:
            val = int(break_match.group(1))
            unit = break_match.group(2)
            duration_ms = val if unit == "ms" else val * 1000
            create_silence(duration_ms, chunk_path, current_sample_rate)
            chunk_files.append(chunk_path)
        else:
            active_settings = settings.copy()
            if "<DOWN_TRIGGER>" in part:
                part = part.replace("<DOWN_TRIGGER>", "")
                active_settings['noise_scale'] = 0.4
                active_settings['noise_w'] = 0.6

            clean_part = re.sub(r'<[^>]+>', '', part)
            if not clean_part: continue

            success = run_piper_cmd(clean_part, chunk_path, active_settings, active_model_path)
            
            if success and os.path.exists(chunk_path):
                chunk_files.append(chunk_path)
                try:
                    with wave.open(chunk_path, 'rb') as w:
                        current_sample_rate = w.getframerate()
                except: pass

    if not chunk_files:
        print("   ❌ No audio generated.")
        return None

    final_path = os.path.join(OUTPUT_WAVS_DIR, final_filename)
    merge_wavs(chunk_files, final_path)
    
    # Cleanup
    for f in chunk_files:
        try: os.remove(f)
        except: pass
        
    print(f"   ✅ Saved: {final_filename}")

    if settings['autoplay']:
        try: winsound.PlaySound(final_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except: pass
    
    return final_path

def run_piper_cmd(text, output_path, settings, model_path):
    if not os.path.exists(model_path):
        print(f"\n❌ Model missing: {model_path}")
        return False
    if not os.path.exists(model_path + ".json"):
        print(f"\n❌ Config missing: {model_path}.json")
        return False

    try:
        cmd = [
            PIPER_BINARY,
            "--model", model_path,
            "--output_file", output_path,
            "--length_scale", str(settings['length_scale']),
            "--noise_scale", str(settings['noise_scale']),
            "--noise_w", str(settings['noise_w']),
            "--sentence_silence", str(settings['sentence_silence'])
        ]
        
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            startupinfo=startupinfo
        )
        process.communicate(input=text.encode('utf-8'))
        return process.returncode == 0
    except Exception as e:
        print(f"\n❌ Piper Error: {e}")
        return False

# --- MENUS ---

def get_safe_filename(text):
    text = re.sub(r'version:\s*[\w\-\.]+', '', text, flags=re.IGNORECASE)
    clean = re.sub(r'<[^>]*>', '', text) 
    clean = re.sub(r'[^\w\s]', '', clean)
    words = clean.split()[:5]
    slug = "_".join(words).lower()[:30]
    if not slug: slug = "output"
    return f"{slug}_{int(time.time()) % 10000}.wav"

def mode_interactive():
    print(f"\n--- ⌨️  Interactive Mode ---")
    print("Tags: <break time='500ms'/>, <down/>, version: v1")
    while True:
        text = input("\n📝 Say: ").strip()
        if not text: continue
        if text.lower() in ["back", "exit", "quit"]: break
        filename = get_safe_filename(text)
        process_text_to_audio(text, filename)

def mode_read_file():
    print(f"\n--- 📄 File Reader ---")
    all_files = sorted(glob.glob(os.path.join(TXT_INPUT_DIR, "*.txt")))
    if not all_files:
        print(f"⚠️ No files in '{TXT_INPUT_DIR}'")
        return
    for i, f in enumerate(all_files):
        print(f"   {i+1}. {os.path.basename(f)}")
    choice = input("\nSelect # (or 'q'): ").strip()
    if choice.lower() == 'q': return
    try:
        idx = int(choice) - 1
        with open(all_files[idx], 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if content:
            out_name = os.path.basename(all_files[idx]).replace(".txt", ".wav")
            process_text_to_audio(content, out_name)
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    ensure_setup()
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n" + "="*40)
        print(f"   🎙️  PIPER TTS MANAGER")
        print(f"   📁 Models Dir: {MODELS_DIR}")
        print("="*40)
        print("1. ⌨️  Type & Speak")
        print("2. 📄 Read File")
        print("3. ⚙️ Reset Settings")
        print("4. 🚪 Exit")
        choice = input("\n> ").strip()
        if choice == "1": mode_interactive()
        elif choice == "2": mode_read_file()
        elif choice == "3": regenerate_default_settings()
        elif choice == "4": sys.exit(0)

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: sys.exit(0)
