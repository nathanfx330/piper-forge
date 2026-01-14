import os
import sys
import shutil
import glob
import datetime
from config import *

# --- CONFIGURATION ---
SOURCE_DIR = "vocoder_checkpoints"
CONFIG_FILE = os.path.join("BigVGAN", "configs", "finetune_22khz.json")
BACKUP_ROOT = "backups_bigvgan"

def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024) # Convert to MB

def do_backup():
    print(f"\n--- 💾 Creating BigVGAN Backup ---")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: Training directory '{SOURCE_DIR}' does not exist.")
        return

    # 1. Prepare Timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Try to find the latest step number for the folder name
    step_num = "unknown"
    checkpoints = glob.glob(os.path.join(SOURCE_DIR, "g_*.pt"))
    if checkpoints:
        latest = max(checkpoints, key=os.path.getmtime)
        import re
        match = re.search(r"g_(\d+)", os.path.basename(latest))
        if match:
            step_num = match.group(1)

    backup_name = f"step{step_num}_{timestamp}"
    destination = os.path.join(BACKUP_ROOT, backup_name)

    if not os.path.exists(BACKUP_ROOT):
        os.makedirs(BACKUP_ROOT)

    # 2. Copy Checkpoints
    print(f"   Source: {SOURCE_DIR}/")
    print(f"   Dest:   {destination}/")
    print("   ⏳ Copying files...")

    try:
        shutil.copytree(SOURCE_DIR, destination)
        
        # 3. Copy Config (Critical for restoring later)
        if os.path.exists(CONFIG_FILE):
            shutil.copy(CONFIG_FILE, os.path.join(destination, "config_backup.json"))
            
        print(f"\n✅ Success! Backup saved as: {backup_name}")
    except Exception as e:
        print(f"❌ Backup failed: {e}")

def do_restore():
    print(f"\n--- ♻️  Restore BigVGAN from Backup ---")
    
    # 1. Find backups
    if not os.path.exists(BACKUP_ROOT):
        print("❌ No 'backups_bigvgan' folder found.")
        return

    backups = sorted(os.listdir(BACKUP_ROOT))
    valid_backups = [b for b in backups if os.path.isdir(os.path.join(BACKUP_ROOT, b))]

    if not valid_backups:
        print("❌ No backups found.")
        return

    # 2. List them
    print("Available Restore Points:")
    for i, b in enumerate(valid_backups):
        path = os.path.join(BACKUP_ROOT, b)
        size = get_dir_size(path)
        print(f"   {i+1}. {b}  ({size:.1f} MB)")

    # 3. User Selection
    choice = input("\nSelect a number to restore (or 'q' to cancel): ").strip()
    if choice.lower() == 'q': return

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(valid_backups):
            print("❌ Invalid selection.")
            return
        
        target_backup = valid_backups[idx]
        src_path = os.path.join(BACKUP_ROOT, target_backup)
    except ValueError:
        print("❌ Invalid input.")
        return

    # 4. SAFETY WARNING
    print(f"\n⚠️  WARNING: This will DELETE the current '{SOURCE_DIR}' folder.")
    print(f"    and replace it with '{target_backup}'.")
    confirm = input("    Type 'yes' to confirm: ").lower().strip()

    if confirm != "yes":
        print("🚫 Restore cancelled.")
        return

    # 5. Perform Restore
    try:
        # Stop user if they left the training script running? 
        # (We can't easily stop it, but we can warn if files are locked)
        
        if os.path.exists(SOURCE_DIR):
            print(f"   🗑️  Deleting current {SOURCE_DIR}...")
            shutil.rmtree(SOURCE_DIR)
        
        print(f"   📂 Restoring from {target_backup}...")
        shutil.copytree(src_path, SOURCE_DIR)
        
        # Restore Config
        backup_config = os.path.join(src_path, "config_backup.json")
        if os.path.exists(backup_config):
            shutil.copy(backup_config, CONFIG_FILE)
            print("   ⚙️  Restored matching config file.")

        print("\n✅ Restore Complete. You can resume training (Script 10).")
    except Exception as e:
        print(f"❌ Restore failed: {e}")
        print("   (Did you leave Script 10 running? Please close it first.)")

def main():
    while True:
        print("\n" + "="*40)
        print("      🛡️  BigVGAN CHECKPOINT MANAGER")
        print("="*40)
        print("1. 💾 Backup Current Vocoder State")
        print("2. ♻️  Restore Old Backup")
        print("3. 🚪 Exit")
        
        choice = input("\nChoose option: ").strip()

        if choice == "1":
            do_backup()
        elif choice == "2":
            do_restore()
        elif choice == "3":
            print("👋 Bye")
            sys.exit(0)
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()