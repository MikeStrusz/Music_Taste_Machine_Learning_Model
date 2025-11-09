import pandas as pd
from config import INTERNAL_RANKINGS_PATH, PUBLIC_RANKINGS_PATH

def export_internal_rankings(df):
    """Exports detailed internal rankings."""
    if df.empty:
        print("✗ No data to export for internal rankings")
        return
        
    internal_columns = [
        "Artist Name", "Album", "Best Song", "Genre", "Release Date",
        "Scrobble Score", "Total Scrobbles", "Avg Per Track", "Obsession Score", "Prediction Score",
        "Similar Artists", "Album Length", "Discovery Method", "Album Art URL",
        "manual_boost", "manual_notes", "final_rank_override"
    ]
    
    # Ensure all columns exist
    for col in internal_columns:
        if col not in df.columns:
            df[col] = ""

    df[internal_columns].to_csv(INTERNAL_RANKINGS_PATH, index=False)
    print(f"✓ Internal rankings exported: {INTERNAL_RANKINGS_PATH}")
    print(f"  - {len(df)} albums")
    print(f"  - Top album: {df.iloc[0]['Artist Name']} - {df.iloc[0]['Album']} (Score: {df.iloc[0]['Scrobble Score']:.2f})")

def export_public_rankings(df):
    """Exports clean public rankings."""
    if df.empty:
        print("✗ No data to export for public rankings")
        return
        
    public_columns = [
        "Rank", "Cover Art", "Artist Name", "Album", "Best Song", "Genre", "Release Date",
        "Mini-Review", "Personal Anecdotes"
    ]
    
    public_df = df.copy()
    
    # Map to public format
    public_df["Cover Art"] = public_df.get("Album Art URL", "")
    public_df["Mini-Review"] = ""
    public_df["Personal Anecdotes"] = ""
    
    # Calculate rank
    public_df["Rank"] = public_df["Scrobble Score"].rank(ascending=False, method='dense').astype(int)
    
    # Select public columns
    public_df = public_df[public_columns]
    public_df.to_csv(PUBLIC_RANKINGS_PATH, index=False)
    
    print(f"✓ Public rankings exported: {PUBLIC_RANKINGS_PATH}")
    print(f"  - {len(public_df)} albums ranked")
    print(f"  - #1: {public_df.iloc[0]['Artist Name']} - {public_df.iloc[0]['Album']}")
