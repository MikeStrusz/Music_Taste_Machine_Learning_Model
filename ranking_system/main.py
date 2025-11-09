import argparse
import data_handler
import scoring_engine
import exporters
from config import INTERNAL_RANKINGS_PATH

def generate_initial_rankings(existing_internal_df=None):
    """Generates initial rankings from scratch."""
    print("\\n=== GENERATING INITIAL RANKINGS ===")
    
    primary_df = data_handler.load_primary_dataset()
    if primary_df.empty:
        print("✗ Cannot generate rankings - no album data!")
        return pd.DataFrame()
        
    scrobbles_df = data_handler.load_scrobbles_data()
    predictions_df = data_handler.load_predictions_data()

    merged_df = data_handler.merge_data(primary_df, scrobbles_df, predictions_df, existing_internal_df)
    
    if merged_df.empty:
        print("✗ No data after merging!")
        return pd.DataFrame()
    
    # Calculate scores
    merged_df = scoring_engine.calculate_obsession_score(merged_df)
    ranked_df = scoring_engine.calculate_scrobble_score(merged_df)

    exporters.export_internal_rankings(ranked_df)
    print("✓ Initial rankings generation complete!")
    return ranked_df

def update_rankings():
    """Updates existing rankings with new data."""
    print("\\n=== UPDATING RANKINGS ===")
    existing_internal_df = data_handler.load_existing_rankings(INTERNAL_RANKINGS_PATH)
    return generate_initial_rankings(existing_internal_df)

def main():
    """Main function to run the ranking system."""
    parser = argparse.ArgumentParser(description="2025 Album Ranking System")
    parser.add_argument("--update-only", action="store_true", help="Update rankings with new albums")
    parser.add_argument("--full-refresh", action="store_true", help="Perform full refresh of rankings")
    parser.add_argument("--export-public", action="store_true", help="Export public rankings format")

    args = parser.parse_args()

    ranked_df = None

    if args.full_refresh:
        ranked_df = generate_initial_rankings()
    elif args.update_only:
        ranked_df = update_rankings()
    else:
        # Default to update
        print("No action specified - performing update...")
        ranked_df = update_rankings()

    if args.export_public:
        if ranked_df is None or ranked_df.empty:
            print("Loading existing rankings for public export...")
            ranked_df = data_handler.load_existing_rankings(INTERNAL_RANKINGS_PATH)
        if not ranked_df.empty:
            exporters.export_public_rankings(ranked_df)
        else:
            print("✗ No data available for public export")

if __name__ == "__main__":
    main()
