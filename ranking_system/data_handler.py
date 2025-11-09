import pandas as pd
import requests
from datetime import datetime
from config import PRIMARY_DATASET_PATH, MODEL_SPLIT_DATE, INTERNAL_RANKINGS_PATH, LASTFM_API_KEY, USERNAME

def load_primary_dataset():
    """Loads the album dataset."""
    try:
        df = pd.read_csv(PRIMARY_DATASET_PATH)
        print(f"✓ Loaded {len(df)} albums from {PRIMARY_DATASET_PATH}")
        return df
    except FileNotFoundError:
        print(f"✗ Error: Album dataset not found at {PRIMARY_DATASET_PATH}")
        return pd.DataFrame()

def fetch_lastfm_scrobbles():
    """Fetches recent scrobbles from Last.fm API."""
    BASE_URL = 'http://ws.audioscrobbler.com/2.0/'
    start_of_2025 = int(datetime(2025, 1, 1).timestamp())
    all_scrobbles = []
    page = 1
    max_pages = 10  # Limit pages for testing

    print("Fetching scrobbles from Last.fm...")
    
    while page <= max_pages:
        params = {
            'method': 'user.getrecenttracks',
            'user': USERNAME,
            'api_key': LASTFM_API_KEY,
            'format': 'json',
            'limit': 200,
            'page': page
        }
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            data = response.json()
            tracks = data.get('recenttracks', {}).get('track', [])

            if not tracks:
                break

            for track in tracks:
                timestamp_str = track.get('date', {}).get('uts', '0')
                timestamp = int(timestamp_str)

                if timestamp >= start_of_2025:
                    all_scrobbles.append({
                        'artist': track.get('artist', {}).get('#text', 'Unknown'),
                        'title': track.get('name', 'Unknown'),
                        'album': track.get('album', {}).get('#text', 'Unknown'),
                        'timestamp': timestamp
                    })
                else:
                    # Stop if we reach older tracks
                    break
            else:
                page += 1
                continue
            break
                
        except Exception as e:
            print(f"✗ Error fetching page {page}: {e}")
            break
            
    scrobbles_df = pd.DataFrame(all_scrobbles)
if not scrobbles_df.empty:
        scrobbles_df = scrobbles_df.rename(columns={'artist': 'Artist Name', 'album': 'Album'})
    
    print(f"✓ Fetched {len(scrobbles_df)} scrobbles from 2025")
    return scrobbles_df

def load_scrobbles_data():
    """Loads fresh scrobbles data."""
    return fetch_lastfm_scrobbles()

def load_predictions_data():
    """Loads predictions data (placeholder)."""
    try:
        df = pd.read_csv("../data/predictions.csv")
        print("✓ Loaded predictions data")
        return df
    except FileNotFoundError:
        print("ℹ No predictions.csv found - continuing without prediction data")
        return pd.DataFrame()

def load_existing_rankings(file_path):
    """Loads existing rankings."""
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded existing rankings with {len(df)} albums")
        return df
    except FileNotFoundError:
        print("ℹ No existing rankings found - starting fresh")
        return pd.DataFrame()

def merge_data(primary_df, scrobbles_df, predictions_df, existing_internal_df=None):
    """Merges all data sources."""
    if primary_df.empty:
        print("✗ No album data to process!")
        return pd.DataFrame()
        
    merged_df = primary_df.copy()
    print(f"Starting with {len(merged_df)} albums")

    # Add this line to ensure track_count exists
merged_df['track_count'] = merged_df.get('Track Count', merged_df.get('track_count', 1))

    # Merge scrobbles data
if not scrobbles_df.empty:
        # Calculate scrobble statistics
        track_counts = scrobbles_df.groupby(["Artist Name", "Album"]).size().reset_index(name="track_count")
        scrobble_totals = scrobbles_df.groupby(["Artist Name", "Album"]).size().reset_index(name="Total Scrobbles")
        
        # Merge with album data
        merged_df = pd.merge(merged_df, track_counts, on=["Artist Name", "Album"], how="left")
        merged_df = pd.merge(merged_df, scrobble_totals, on=["Artist Name", "Album"], how="left")
        
        # Calculate average per track
        merged_df["Avg Per Track"] = merged_df["Total Scrobbles"] / merged_df["track_count"]
        merged_df = merged_df.fillna(0)
        print(f"✓ Merged scrobbles data for {len(scrobbles_df)} tracks")

    # Ensure required scoring columns exist
    required_cols = {
        "Total Scrobbles": 0,
        "Avg Per Track": 0.0, 
        "track_count": 0,
        "Prediction Score": 0.0,
        "Obsession Score": 0.0
    }
    
    for col, default in required_cols.items():
        if col not in merged_df.columns:
            merged_df[col] = default

    # Handle manual columns from existing rankings
    if existing_internal_df is not None and not existing_internal_df.empty:
        manual_cols = ["manual_boost", "manual_notes", "final_rank_override"]
        for col in manual_cols:
            if col in existing_internal_df.columns:
                merged_df = pd.merge(
                    merged_df, 
                    existing_internal_df[["Artist Name", "Album", col]], 
                    on=["Artist Name", "Album"], 
                    how="left", 
                    suffixes=("", "_existing")
                )
                merged_df[col] = merged_df[f"{col}_existing"].combine_first(merged_df[col])
                merged_df = merged_df.drop(columns=[f"{col}_existing"])
        print("✓ Preserved manual edits from existing rankings")

    # Filter by release date if available
    if "Release Date" in merged_df.columns:
        try:
            merged_df["Release Date"] = pd.to_datetime(merged_df["Release Date"], errors='coerce')
            model_split_datetime = pd.to_datetime(MODEL_SPLIT_DATE)
            initial_count = len(merged_df)
            merged_df = merged_df[merged_df["Release Date"] >= model_split_datetime]
            print(f"✓ Filtered to {len(merged_df)} albums released after {MODEL_SPLIT_DATE} (from {initial_count})")
        except Exception as e:
            print(f"ℹ Could not filter by release date: {e}")

    return merged_df
