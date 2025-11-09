
# 2025 VALIDATION SCRIPT
# Run this after training your model on clean data

import pandas as pd
import os

def validate_2025_predictions(model, feature_pipeline):
    """
    Validate your model on 2025 data that was NOT in training
    """
    print(" RUNNING 2025 VALIDATION")
    print("="*50)
    
    # Load 2025 validation albums
    validation_df = pd.read_csv('./data_clean_2025_validation/validation_2025_albums.csv')
    print(f"Validation albums loaded: {len(validation_df)}")
    
    # TODO: Add your prediction code here
    # This should be similar to your weekly prediction pipeline
    
    # Example structure:
    # 1. Get features for 2025 albums
    # 2. Run model predictions
    # 3. Compare with actual listening (from your 2025.csv)
    
    # Load actual 2025 listening
    actual_2025 = pd.read_csv('C:/Users/mrstr/Downloads/2025.csv')
    actual_albums = actual_2025[['Album Name', 'Artist Name(s)']].drop_duplicates()
    print(f"Actual albums listened to: {len(actual_albums)}")
    
    # Calculate accuracy
    # (You'll need to implement the prediction part)
    
    return validation_results

# Run validation
# results = validate_2025_predictions(your_trained_model, your_feature_pipeline)
