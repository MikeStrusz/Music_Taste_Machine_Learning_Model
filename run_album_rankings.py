"""
2025 Album Ranking System - Run Script
Just run this script to start your album rankings!
"""

import os
import subprocess
import sys

def main():
    print("🎵 2025 Album Ranking System 🎵")
    print("=" * 40)
    
    # Check if we're in the right directory
    current_dir = os.getcwd()
    ranking_system_dir = r"C:\Users\mrstr\Downloads\9_Module_Tableau\Capstone\Music_Taste_Machine_Learning_Model\ranking_system"
    
    if current_dir != ranking_system_dir:
        print(f"Changing to ranking system directory...")
        os.chdir(ranking_system_dir)
    
    print("Ready to run! Choose an option:")
    print("1. Full refresh + public export (recommended for first run)")
    print("2. Update only (preserves manual edits)")
    print("3. Export public rankings only")
    print("4. Test system")
    
    choice = input("\\nEnter your choice (1-4): ").strip()
    
    commands = {
        '1': 'python main.py --full-refresh --export-public',
        '2': 'python main.py --update-only --export-public', 
        '3': 'python main.py --export-public',
        '4': 'python main.py --full-refresh'
    }
    
    if choice in commands:
        print(f"\\nRunning: {commands[choice]}")
        print("-" * 40)
        result = subprocess.run(commands[choice], shell=True)
        if result.returncode == 0:
            print("\\n✅ Success! Check the outputs folder for your rankings.")
        else:
            print("\\n❌ Something went wrong. Check the error messages above.")
    else:
        print("Invalid choice. Running full refresh with public export...")
        subprocess.run('python main.py --full-refresh --export-public', shell=True)

if __name__ == "__main__":
    main()
