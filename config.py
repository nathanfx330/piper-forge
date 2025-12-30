# config.py

# --- VOICE IDENTITY ---
# Change this to name your model (e.g., "dion", "narrator", "jarvis")
VOICE_NAME = "my_custom_voice" 

# --- FOLDER STRUCTURE ---
# Put your long .mp3/.wav files here
RAW_AUDIO_DIR = "raw_audio" 
# Where the sliced clips and metadata.csv will go
DATASET_DIR = "dataset" 
# Where the actual training happens (checkpoints)
TRAINING_DIR = "training_checkpoints" 
# Where the final usable model files will be saved
OUTPUT_DIR = "final_models" 
# Path to the piper engine folder
PIPER_DIR = "piper" 

# --- AUDIO SETTINGS ---
# 22050 is standard for Piper Medium/High. 
SAMPLE_RATE = 22050 
# "en-us" for American, "en-gb" for British
LANGUAGE_CODE = "en-us" 

# --- TRAINING HYPERPARAMETERS ---
# "medium" = Fast, Robust (Recommended). "high" = Better quality, slower, requires cleaner audio.
QUALITY = "medium" 
# Batch Size: 32 for Medium (RTX 3060+), 16 or 8 if you run out of VRAM.
BATCH_SIZE = 8 
# How often to save a checkpoint (in epochs)
SAVE_EVERY_EPOCHS = 20 
# Total epochs (you can stop earlier manually)
MAX_EPOCHS = 6000 

# --- BASE MODEL (Transfer Learning) ---
# We download this once to start training on top of it.
# Current: HFC Female Medium (Good general purpose American female)
BASE_MODEL_URL = "https://huggingface.co/rhasspy/piper-checkpoints/resolve/main/en/en_US/hfc_female/medium/en_US-hfc_female-medium.ckpt"
BASE_MODEL_FILENAME = "base_model.ckpt"