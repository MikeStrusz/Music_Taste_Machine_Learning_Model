import pandas as pd
from config import POST_MODEL_WEIGHTS, PRE_MODEL_WEIGHTS, MODEL_SPLIT_DATE

def calculate_obsession_score(df):
    """Calculates obsession score based on listening patterns."""
    if df.empty:
        return df
        
    # Simple implementation using total scrobbles as proxy
    if "Total Scrobbles" in df.columns:
        df["Obsession Score"] = (df["Total Scrobbles"] / 10).clip(upper=100)
    else:
        df["Obsession Score"] = 0
        
    print("✓ Calculated obsession scores")
    return df

def calculate_scrobble_score(df):
    """Calculates final scrobble scores."""
    if df.empty:
        return df
        
    df["Scrobble Score"] = 0.0
    
    # Ensure Release Date is datetime
    try:
        df["Release Date"] = pd.to_datetime(df["Release Date"], errors='coerce')
        model_split_datetime = pd.to_datetime(MODEL_SPLIT_DATE)
    except:
        print("ℹ Using default scoring (could not parse dates)")
        # Apply default scoring if dates aren't available
        df["Scrobble Score"] = df["Total Scrobbles"] * 0.5 + df["Obsession Score"] * 0.5
        return df.sort_values("Scrobble Score", ascending=False)

    # Apply Post-Model formula (Feb 2025+)
    post_model_mask = df["Release Date"] >= model_split_datetime
    if post_model_mask.any():
        A, B, C, D = POST_MODEL_WEIGHTS['A'], POST_MODEL_WEIGHTS['B'], POST_MODEL_WEIGHTS['C'], POST_MODEL_WEIGHTS['D']
        df.loc[post_model_mask, "Scrobble Score"] = (
            df.loc[post_model_mask, "Total Scrobbles"] * A +
            df.loc[post_model_mask, "Avg Per Track"] * df.loc[post_model_mask, "track_count"] * B +
            df.loc[post_model_mask, "Obsession Score"] * C +
            df.loc[post_model_mask, "Prediction Score"] * D
        )
        print(f"✓ Applied post-model scoring to {post_model_mask.sum()} albums")

    # Apply Pre-Model formula (Before Feb 2025)
    pre_model_mask = df["Release Date"] < model_split_datetime
    if pre_model_mask.any():
        X, Y, Z = PRE_MODEL_WEIGHTS['X'], PRE_MODEL_WEIGHTS['Y'], PRE_MODEL_WEIGHTS['Z']
        df.loc[pre_model_mask, "Scrobble Score"] = (
            df.loc[pre_model_mask, "Total Scrobbles"] * X +
            df.loc[pre_model_mask, "Avg Per Track"] * df.loc[pre_model_mask, "track_count"] * Y +
            df.loc[pre_model_mask, "Obsession Score"] * Z
        )
        print(f"✓ Applied pre-model scoring to {pre_model_mask.sum()} albums")

    # Sort by score
    df = df.sort_values("Scrobble Score", ascending=False)
    return df
