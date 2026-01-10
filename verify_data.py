
import os
import glob
import sys

def verify_app_data():
    required_directories = [
        'data',
        'predictions',
        'feedback'
    ]

    required_data_files = [
        'data/nuked_albums.csv',
        'data/nmf_album_covers.csv',
        'data/nmf_album_links.csv',
        'data/nmf_similar_artists.csv',
        'data/liked_artists_only_similar.csv',
        'data/df_cleaned_pre_standardized.csv'
    ]

    missing_items = []

    print("\n--- Verifying Streamlit App Data Dependencies ---")

    # Check directories
    for d in required_directories:
        if not os.path.exists(d):
            missing_items.append(f"Missing directory: {d}")
        else:
            print(f"Found directory: {d}")

    # Check specific data files
    for f in required_data_files:
        if not os.path.exists(f):
            missing_items.append(f"Missing data file: {f}")
        else:
            print(f"Found data file: {f}")

    # Check for at least one prediction file
    prediction_files = glob.glob('predictions/*_Album_Recommendations.csv')
    if not prediction_files:
        missing_items.append("No prediction files found in 'predictions/' (e.g., 'predictions/MM-DD-YY_Album_Recommendations.csv')")
    else:
        print(f"Found {len(prediction_files)} prediction files in 'predictions/'")

    if missing_items:
        print("\n--- Verification FAILED ---")
        for item in missing_items:
            print(f"ERROR: {item}")
        print("\nPlease ensure your Jupyter Notebook has been run to generate all necessary data files and directories.")
        sys.exit(1) # Exit with an error code
    else:
        print("\n--- Verification SUCCESS: All required data dependencies are present! ---")
        return True

if __name__ == '__main__':
    verify_app_data()
