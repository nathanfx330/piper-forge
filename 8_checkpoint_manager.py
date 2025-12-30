import os
import sys
import shutil
import glob
import datetime
from config import *

# Folder where we store the zip/copies
BACKUP_ROOT = "backups"

def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024) # Convert to MB

def do_backup():
    print(f"\n--- 💾 Creating Backup ---")
    
    if not os.path.exists(TRAINING_DIR):
        print(f"❌ Error: Training directory '{TRAINING_DIR}' does not exist.")
        return

    # 1. Prepare Timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_name = f"backup_{timestamp}"
    destination = os.path.join(BACKUP_ROOT, backup_name)

    if not os.path.exists(BACKUP_ROOT):
        os.makedirs(BACKUP_ROOT)

    # 2. Copy
    print(f"   Source: {TRAINING_DIR}/")
    print(f"   Dest:   {destination}/")
    print("   ⏳ Copying files (this may take a moment)...")

    try:
        shutil.copytree(TRAINING_DIR, destination)
        print(f"\n✅ Success! Backup saved to:\n   {destination}")
    except Exception as e:
        print(f"❌ Backup failed: {e}")

def do_restore():
    print(f"\n--- ♻️  Restore from Backup ---")
    
    # 1. Find backups
    if not os.path.exists(BACKUP_ROOT):
        print("❌ No 'backups' folder found.")
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
    print(f"\n⚠️  WARNING: This will DELETE the current '{TRAINING_DIR}' folder.")
    print(f"    and replace it with '{target_backup}'.")
    confirm = input("    Type 'yes' to confirm: ").lower().strip()

    if confirm != "yes":
        print("🚫 Restore cancelled.")
        return

    # 5. Perform Restore
    try:
        if os.path.exists(TRAINING_DIR):
            print(f"   🗑️  Deleting current {TRAINING_DIR}...")
            shutil.rmtree(TRAINING_DIR)
        
        print(f"   📂 Restoring from {target_backup}...")
        shutil.copytree(src_path, TRAINING_DIR)
        print("\n✅ Restore Complete. You can run 4_train.py to resume.")
    except Exception as e:
        print(f"❌ Restore failed: {e}")

def main():
    while True:
        print("\n" + "="*40)
        print("      🛡️  CHECKPOINT MANAGER")
        print("="*40)
        print("1. 💾 Backup Current Training")
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