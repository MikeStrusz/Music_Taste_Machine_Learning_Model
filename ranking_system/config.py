# config.py

# API Credentials
LASTFM_API_KEY = "74a510ecc9fc62bf3e0edc6adc2e99f9"
USERNAME = "Strusz_Music"

# Scoring Weights (Post-Model Albums - Feb 2025+)
POST_MODEL_WEIGHTS = {
    'A': 0.4,  # Total scrobbles weight
    'B': 0.3,  # Avg per track * track count weight  
    'C': 0.2,  # Obsession score weight
    'D': 0.1   # Prediction score weight
}

# Scoring Weights (Pre-Model Albums - Before Feb 2025)
PRE_MODEL_WEIGHTS = {
    'X': 0.6,  # Total scrobbles weight (higher without predictions)
    'Y': 0.3,  # Avg per track * track count weight
    'Z': 0.1   # Obsession score weight
}

# File Paths (relative to ranking_system folder)
PRIMARY_DATASET_PATH = "../data/2025_albums.csv"
INTERNAL_RANKINGS_PATH = "../outputs/2025_internal_rankings.csv"
PUBLIC_RANKINGS_PATH = "../outputs/2025_public_rankings.csv"

# Other configurations
MODEL_SPLIT_DATE = "2025-02-01"
