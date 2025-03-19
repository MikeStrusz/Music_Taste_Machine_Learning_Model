# feedback_backup.py
import pandas as pd
import os
import shutil
from datetime import datetime

def backup_feedback_files() :
    """
    Create a timestamped backup of feedback files
    """
    # Create timestamp for the backup folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"feedback_backup_{timestamp}"
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    
    # Files to backup
    files_to_backup = [
        "feedback/feedback.csv",
        "feedback/public_feedback.csv",
        "data/nmf_album_links.csv",
        "data/nmf_album_covers.csv"
    ]
    
    # Backup each file if it exists
    backed_up_files = []
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            # Create directory structure in backup folder if needed
            backup_file_dir = os.path.dirname(os.path.join(backup_dir, file_path))
            os.makedirs(backup_file_dir, exist_ok=True)
            
            # Copy the file
            shutil.copy2(file_path, os.path.join(backup_dir, file_path))
            backed_up_files.append(file_path)
            print(f"Backed up: {file_path}")
        else:
            print(f"File not found, skipping: {file_path}")
    
    # Create a summary file
    with open(os.path.join(backup_dir, "backup_info.txt"), "w") as f:
        f.write(f"Backup created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Files backed up:\n")
        for file in backed_up_files:
            f.write(f"- {file}\n")
    
    print(f"\nBackup completed! Files saved to: {backup_dir}")
    print(f"Total files backed up: {len(backed_up_files)}")
    
    return backup_dir

if __name__ == "__main__":
    print("Starting feedback backup...")
    backup_dir = backup_feedback_files()
    print(f"\nTo restore these files later, copy them from {backup_dir} back to their original locations.")
