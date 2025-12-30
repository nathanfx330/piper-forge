import os
import glob
import librosa
import soundfile as sf
import whisper
import warnings
import torch
from tqdm import tqdm
from config import *  # Imports paths from config.py

# Ignore librosa warnings
warnings.filterwarnings("ignore")

def check_raw_files():
    files = glob.glob(os.path.join(RAW_AUDIO_DIR, "*"))
    # Filter for audio extensions
    audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg'))]
    
    if not audio_files:
        print(f"❌ ERROR: No audio files found in '{RAW_AUDIO_DIR}/'")
        print("   Please put your recordings there first.")
        return []
    return audio_files

def main():
    print(f"--- 🔪 Audio Slicer & Transcriber for '{VOICE_NAME}' ---")
    
    # 1. Check Input
    raw_files = check_raw_files()
    if not raw_files:
        return

    # 2. Setup Output Paths
    wavs_dir = os.path.join(DATASET_DIR, "wavs")
    if not os.path.exists(wavs_dir):
        os.makedirs(wavs_dir)
    
    metadata_path = os.path.join(DATASET_DIR, "metadata.csv")

    # 3. Load Whisper
    print("\n🧠 Loading Whisper Model (large)...")
    # 'medium' is a good balance. Use 'large' if you have 12GB+ VRAM and want perfection.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("large", device=device)

    metadata_lines = []
    global_count = 0

    # 4. Process Files
    print(f"\n🚀 Processing {len(raw_files)} file(s)...")
    
    for raw_file in raw_files:
        print(f"   Reading: {os.path.basename(raw_file)}")
        
        try:
            # Load and Resample to 22050Hz Mono
            y, sr = librosa.load(raw_file, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"   ⚠️  Skipping corrupt file: {e}")
            continue

        # Slice on silence (Top DB 40 is standard for clean speech)
        intervals = librosa.effects.split(y, top_db=40, frame_length=2048, hop_length=512)
        
        print(f"   -> Found {len(intervals)} potential segments.")

        for start, end in tqdm(intervals, desc="      Transcribing", leave=False):
            chunk = y[start:end]
            duration = len(chunk) / sr
            
            # Filter length (Piper hates < 1s and > 10s)
            if duration < 1.0 or duration > 10.0:
                continue
            
            global_count += 1
            
            # Naming format: voice_name_0001.wav
            filename = f"{VOICE_NAME}_{global_count:04d}.wav"
            filepath = os.path.join(wavs_dir, filename)
            
            # Save as 16-bit PCM WAV (Crucial for Piper)
            sf.write(filepath, chunk, SAMPLE_RATE, subtype='PCM_16')
            
            # Transcribe
            result = model.transcribe(filepath, language="en")
            text = result['text'].strip().replace("\n", " ")
            
            # Hallucination Filters
            bad_starts = ["Subtitle", "Copyright", "Translated", "Captioning"]
            if any(text.startswith(b) for b in bad_starts) or len(text) < 2:
                os.remove(filepath) # Delete bad audio
                global_count -= 1   # Reset count
                continue
                
            # Add to metadata list
            metadata_lines.append(f"{filename}|{text}")

    # 5. Write Metadata
    print(f"\n💾 Saving {metadata_path}...")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(metadata_lines))

    print(f"\n--- ✅ Dataset Ready ---")
    print(f"Total Clips: {len(metadata_lines)}")
    print(f"Location: {DATASET_DIR}/")
    print("Action: Check metadata.csv quickly to ensure text looks correct.")
    print("Next Step: Run Script 4 to prepare training tensors.")

if __name__ == "__main__":
    main()