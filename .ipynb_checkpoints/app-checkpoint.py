import streamlit as st

st.set_page_config(
    page_title="New Music Friday Regression Model",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

from verify_data import verify_app_data

# Run data verification at the start
verify_app_data()
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import glob
import os
import shutil
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from typing import Dict
import networkx as nx 
import random 

# Check if App is Running Locally or on Streamlit's Servers
def is_running_on_streamlit():
    return os.getenv("STREAMLIT_SERVER_RUNNING", "False").lower() == "true"

# Use this flag to control feedback buttons
IS_LOCAL = not is_running_on_streamlit()

@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_hidden_gems_cache():
    """Load pre-computed Hidden Gems"""
    cache_file = 'data/hidden_gems_cache.csv'
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    return pd.DataFrame()

@st.cache_data(ttl=86400)
def load_hidden_gems_sample():
    """Load sample of Hidden Gems for instant display"""
    sample_file = 'data/hidden_gems_sample.csv'
    if os.path.exists(sample_file):
        return pd.read_csv(sample_file)
    return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_all_albums(prediction_files, months_back=12):
    """Load all albums from recent months (cached for performance)"""
    all_albums_list = []
    max_date = max(date_obj for _, _, date_obj in prediction_files)
    cutoff_date = max_date - timedelta(days=30*months_back)
    
    for date, file, date_obj in prediction_files:
        if date_obj < cutoff_date:
            continue
        
        # Direct file reading instead of calling load_predictions()
        try:
            df = pd.read_csv(file)
            
            # Standardize column names
            if 'Album' in df.columns:
                df['Album Name'] = df['Album']
            elif 'Album Name' in df.columns:
                df['Album'] = df['Album Name']
            
            if 'Artist Name(s)' in df.columns:
                df['Artist'] = df['Artist Name(s)']
            
            if 'Predicted_Score' in df.columns:
                df['avg_score'] = df['Predicted_Score']
            
            if 'playlist_origin' not in df.columns:
                df['playlist_origin'] = 'unknown'
            
            if len(df) > 0:
                df['source_week'] = date
                df['source_date'] = date_obj
                all_albums_list.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    if not all_albums_list:
        return pd.DataFrame()
    
    combined = pd.concat(all_albums_list, ignore_index=True)
    return combined

@st.cache_data
def calculate_exact_playtime(album_name, artist):
    """
    Calculate exact playtime by summing track durations from training data
    """
    try:
        # Load training data (cached)
        training_file = 'data/2026_training_complete_with_features.csv'
        if os.path.exists(training_file):
            training_df = pd.read_csv(training_file)
            
            # Find tracks for this album
            album_tracks = training_df[
                (training_df['Album Name'] == album_name) &
                (training_df['Artist Name(s)'].str.contains(artist, na=False))
            ]
            
            if len(album_tracks) == 0:
                return None, 0
            
            # Sum durations if available
            if 'Duration_ms' in album_tracks.columns:
                total_ms = album_tracks['Duration_ms'].sum()
            elif 'Duration (ms)' in album_tracks.columns:
                total_ms = album_tracks['Duration (ms)'].sum()
            else:
                return None, len(album_tracks)
            
            # Convert to hours:minutes:seconds
            total_seconds = total_ms / 1000
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            
            if hours > 0:
                playtime_str = f"{hours}:{minutes:02d}:{seconds:02d}"
            else:
                playtime_str = f"{minutes}:{seconds:02d}"
            
            return playtime_str, len(album_tracks)
        
        return None, 0
    except Exception as e:
        print(f"Error calculating playtime: {e}")
        return None, 0

@st.cache_data
def load_top_100_artists():
    """Cache the Top 100 artists list"""
    top_artists = set()
    
    try:
        # Method 1: From training data
        training_file = 'data/2026_training_complete_with_features.csv'
        if os.path.exists(training_file):
            df = pd.read_csv(training_file)
            top100_df = df[df['source_type'] == 'top_100_ranked']
            
            for artist in top100_df['Artist Name(s)'].dropna().unique():
                primary = str(artist).split(',')[0].split(';')[0].split(' feat.')[0].strip()
                if primary:
                    top_artists.add(primary)
        
        # Method 2: From your Top 100 CSV if exists
        top100_csv = 'data/top_100_all_years_clean.csv'
        if os.path.exists(top100_csv):
            df = pd.read_csv(top100_csv)
            if 'Artist' in df.columns:
                for artist in df['Artist'].dropna():
                    primary = str(artist).split(',')[0].split(';')[0].split(' feat.')[0].strip()
                    if primary:
                        top_artists.add(primary)
        
        return top_artists
        
    except Exception as e:
        print(f"Error loading Top 100 artists: {e}")
        return set()

# Load globally
TOP_100_ARTISTS = load_top_100_artists()

def highlight_top_100_in_similar(similar_artists_str, top_100_artists):
    """
    Bold artists that appear in your Top 100
    """
    if pd.isna(similar_artists_str) or not similar_artists_str:
        return "No similar artists data"
    
    similar_list = [artist.strip() for artist in similar_artists_str.split(',')]
    highlighted = []
    
    for artist in similar_list:
        # Clean for comparison
        artist_clean = artist.split(',')[0].split(';')[0].split(' feat.')[0].strip()
        
        # Check exact match first
        if artist_clean in top_100_artists:
            highlighted.append(f"**{artist}** ⭐")  # Star for Top 100
        else:
            # Check fuzzy match
            is_top_100 = False
            for top_artist in top_100_artists:
                if artist_clean.lower() == top_artist.lower():
                    highlighted.append(f"**{artist}** ⭐")
                    is_top_100 = True
                    break
                elif artist_clean.lower() in top_artist.lower() or top_artist.lower() in artist_clean.lower():
                    highlighted.append(f"**{artist}** ⭐")
                    is_top_100 = True
                    break
            
            if not is_top_100:
                highlighted.append(artist)
    
    return ', '.join(highlighted)

@st.cache_data
def load_nuked_albums():
    """
    Load the list of nuked albums from the CSV file.
    """
    nuked_albums_file = 'data/nuked_albums.csv'
    if os.path.exists(nuked_albums_file):
        return pd.read_csv(nuked_albums_file)
    return pd.DataFrame(columns=['Artist', 'Album Name', 'Reason'])

# Custom CSS for both notebook content and general styling
st.markdown("""
    <style>
    .notebook-content {
        text-align: left;
        margin-left: 0px;
        padding-left: 0px;
        width: 100%;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        background-color: #ffffff;
    }
    .similar-artists {
        font-style: italic;
        color: #666;
        margin-top: 5px;
    }
    .stMarkdown {
        text-align: left !important;
    }
    .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .spotify-button {
        background-color: #f8f9fa;
        color: #1e1e1e;
        padding: 8px 16px;
        border-radius: 4px;
        text-decoration: none;
        font-size: 0.9rem;
        display: inline-block;
        margin-top: 10px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .spotify-button:hover {
        background-color: #ffffff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-1px);
        text-decoration: none;
        color: #1e1e1e;
    }
    .album-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
    .public-rating-buttons {
        display: flex;
        gap: 5px;
        margin-top: 5px;
        flex-wrap: nowrap;  /* Changed from wrap to nowrap */
        justify-content: space-between;  /* Distribute buttons evenly */
    }

    .public-rating-buttons button {
        flex: 1;
        min-width: 40px;  /* Reduced from 60px for better mobile fit */
        padding: 8px 12px;
        font-size: 0.9rem;
    }

    /* Light gray background for username input */
    .stTextInput>div>div>input {
        background-color: #f8f9fa !important;  /* Light gray */
        border-radius: 4px;
        padding: 8px;
    }

    /* Light gray background for feedback buttons */
    .stButton>button {
        background-color: #f8f9fa !important;  /* Light gray */
        border: 1px solid #e0e0e0 !important;  /* Light gray border */
        color: #333 !important;  /* Darker text for better contrast */
        border-radius: 4px;
        transition: all 0.3s ease;
    }

    /* Hover effect for feedback buttons */
    .stButton>button:hover {
        background-color: #e9ecef !important;  /* Slightly darker gray on hover */
        border-color: #ced4da !important;  /* Darker gray border on hover */
        color: #000 !important;  /* Black text on hover for better contrast */
    }

    /* Light gray background for the review text area */
    .stTextArea>div>div>textarea {
        background-color: #f8f9fa !important;  /* Light gray */
        border-radius: 4px;
        padding: 8px;
    }

    /* Light gray background for the feedback section container */
    .feedback-container {
        background-color: #f8f9fa !important;  /* Light gray */
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
    }

    .gut-score-container {
        background-color: #e8f5e9 !important;  /* Light green background */
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
        border: 1px solid #c8e6c9;
    }

    .gut-score-input {
        background-color: white !important;
        border: 1px solid #4caf50 !important;
    }

    .gut-score-button {
        background-color: #4caf50 !important;
        color: white !important;
        border: none !important;
    }

    .gut-score-button:hover {
        background-color: #388e3c !important;
    }

    .archive-selector {
        margin-bottom: 20px;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    .archive-button {
        background-color: #f8f9fa;
        color: #1e1e1e;
        padding: 5px 10px;
        border-radius: 4px;
        text-decoration: none;
        font-size: 0.9rem;
        border: 1px solid rgba(0, 0, 0, 0.1);
        margin-right: 5px;
    }
    .archive-button:hover {
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def get_all_prediction_files():
    """
    Get all prediction files and their corresponding dates.
    """
    prediction_files = glob.glob('predictions/*_Album_Recommendations.csv')
    if not prediction_files:
        st.error("No prediction files found!")
        return []
    
    # Sort files by date (newest first)
    file_dates = []
    for file in prediction_files:
        date_str = os.path.basename(file).split('_')[0]
        try:
            date_obj = datetime.strptime(date_str, '%m-%d-%y')
            formatted_date = date_obj.strftime('%B %d, %Y')
            file_dates.append((file, date_obj, formatted_date))
        except ValueError:
            # Skip files with invalid date format
            continue
    
    # Sort by date (newest first)
    file_dates.sort(key=lambda x: x[1], reverse=True)
    return file_dates

@st.cache_data
def load_predictions(file_path=None):
    """
    Load the predictions data from a specific file or the latest file if none specified.
    """
    if file_path is None:
        prediction_files = glob.glob('predictions/*_Album_Recommendations.csv')
        if not prediction_files:
            st.error("No prediction files found!")
            return None
        
        file_path = max(prediction_files)
    
    predictions_df = pd.read_csv(file_path)
    
    # Standardize column names - ALWAYS create both columns for compatibility
    if 'Album' in predictions_df.columns:
        predictions_df['Album Name'] = predictions_df['Album']
    elif 'Album Name' in predictions_df.columns:
        predictions_df['Album'] = predictions_df['Album Name']
    
    # Ensure 'Artist' is the primary column name
    if 'Artist Name(s)' in predictions_df.columns:
        predictions_df['Artist'] = predictions_df['Artist Name(s)']
    
    # Map Predicted_Score to avg_score for backward compatibility
    if 'Predicted_Score' in predictions_df.columns:
        predictions_df['avg_score'] = predictions_df['Predicted_Score']
    
    # Ensure 'playlist_origin' column exists (silently add if missing)
    if 'playlist_origin' not in predictions_df.columns:
        predictions_df['playlist_origin'] = 'unknown'  # Default value
    
    # Remove duplicate albums if any
    subset_cols = []
    if 'Artist' in predictions_df.columns: subset_cols.append('Artist')
    if 'Album Name' in predictions_df.columns: subset_cols.append('Album Name')
    
    if subset_cols:
        predictions_df = predictions_df.drop_duplicates(subset=subset_cols, keep='first')
    
    date_str = os.path.basename(file_path).split('_')[0]
    analysis_date = datetime.strptime(date_str, '%m-%d-%y').strftime('%Y-%m-%d')
    
    return predictions_df, analysis_date

@st.cache_data
def load_album_covers():
    try:
        # DON'T RENAME - keep original column names
        return pd.read_csv('data/nmf_album_covers.csv')
    except Exception as e:
        st.error(f"Error loading album covers data: {e}")
        return pd.DataFrame(columns=['Artist', 'Album Name', 'Album Art'])

@st.cache_data
def load_album_links():
    try:
        df = pd.read_csv('data/nmf_album_links.csv')
        # Standardize columns immediately upon loading
        if 'Artist Name(s)' in df.columns:
            df = df.rename(columns={'Artist Name(s)': 'Artist'})
        return df
    except Exception as e:
        st.error(f"Error loading album links data: {e}")
        return pd.DataFrame(columns=['Album Name', 'Artist', 'Spotify URL'])

@st.cache_data
def load_similar_artists():
    try:
        return pd.read_csv('data/nmf_similar_artists.csv')
    except Exception as e:
        st.error(f"Error loading similar artists data: {e}")
        return pd.DataFrame(columns=['Artist', 'Similar Artists'])

@st.cache_data
def load_similar_artists():
    try:
        # Use the file created by your Jupyter notebook
        return pd.read_csv('data/liked_artists_only_similar.csv')
    except Exception as e:
        st.error(f"Error loading similar artists data: {e}")
        return pd.DataFrame(columns=['Artist', 'Similar Artists'])

def load_training_data():
    df = pd.read_csv('data/df_cleaned_pre_standardized.csv')
    return df[df['playlist_origin'] != 'df_nmf'].copy()

def save_gut_score(album_name, artist, score, file_path, notes=""):
    """
    CLEAN VERSION: Only saves when there's an actual score (0-100)
    Now also saves notes to the predictions CSV
    """
    try:
        # Only save if score is valid (0-100)
        if pd.isna(score) or score < 0 or score > 100:
            st.warning(f"⚠️ Invalid score: {score}. Must be 0-100.")
            return
        
        st.write(f"💾 Saving gut score: {artist} - {album_name} → {score}")
        
        # Get date from filename
        source_filename = os.path.basename(file_path)
        date_str = source_filename.split('_')[0]
        
        # ===== 1. SAVE TO PREDICTIONS CSV =====
        try:
            # Load the original predictions file
            predictions_df = pd.read_csv(file_path)
            
            # Find and update the row in the predictions file
            mask = (
                (predictions_df['Artist'] == artist) & 
                (predictions_df['Album Name'] == album_name)
            )
            
            if mask.any():
                # Update both gut_score and notes
                predictions_df.loc[mask, 'gut_score'] = float(score)
                predictions_df.loc[mask, 'gut_score_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                predictions_df.loc[mask, 'notes'] = notes if notes else ""
                
                # Save back to the predictions file
                predictions_df.to_csv(file_path, index=False)
                st.write(f"📄 Updated predictions file: {source_filename}")
            else:
                st.warning(f"⚠️ Album not found in {source_filename}")
        except Exception as e:
            st.error(f"❌ Error updating predictions file: {str(e)}")
        
        # ===== 2. SAVE TO WEEKLY FEEDBACK FILE =====
        weekly_file = f'feedback/{date_str}_gut_scores.csv'
        os.makedirs('feedback', exist_ok=True)
        
        # Create new entry with notes
        new_entry = pd.DataFrame([{
            'Album': album_name,
            'Artist': artist,
            'gut_score': float(score),
            'gut_score_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source_file': source_filename,
            'predicted_score': None,  # Will be filled from predictions file
            'notes': notes if notes else ""
        }])
        
        # Update or create weekly file
        if os.path.exists(weekly_file):
            weekly_df = pd.read_csv(weekly_file)
            
            # Remove any existing entry for this album
            mask = ~((weekly_df['Album'] == album_name) & (weekly_df['Artist'] == artist))
            weekly_df = weekly_df[mask]
            
            # Add new entry
            weekly_df = pd.concat([weekly_df, new_entry], ignore_index=True)
        else:
            weekly_df = new_entry
        
        weekly_df.to_csv(weekly_file, index=False)
        st.write(f"📁 Saved to weekly: {weekly_file}")
        
        # ===== 3. SAVE TO MASTER FILE =====
        master_file = 'feedback/master_gut_scores.csv'
        
        if os.path.exists(master_file):
            master_df = pd.read_csv(master_file)
            
            # Remove any existing entry for this album
            mask = ~((master_df['Album'] == album_name) & (master_df['Artist'] == artist))
            master_df = master_df[mask]
            
            # Add new entry
            master_df = pd.concat([master_df, new_entry], ignore_index=True)
        else:
            master_df = new_entry
        
        master_df.to_csv(master_file, index=False)
        st.write(f"📁 Saved to master: {master_file}")
        
        # Also store in session state for immediate feedback
        album_key = f"{artist}_{album_name}"
        if 'recent_ratings' not in st.session_state:
            st.session_state.recent_ratings = {}
        st.session_state.recent_ratings[album_key] = score
        
        st.success(f"✅ Gut Score {score} saved!")
        
        # Force immediate rerun to show updated feedback
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error saving gut score: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def display_album_predictions(filtered_data, album_covers_df, similar_artists_df, current_file_path=None):
    # ADD THIS LINE - it forces Streamlit to treat each call as unique
    st.markdown("<!-- Unique render -->", unsafe_allow_html=True)
    
    # Initialize session state for real-time feedback
    if 'recent_ratings' not in st.session_state:
        st.session_state.recent_ratings = {}
    
    try:
        album_links_df = load_album_links()
    except Exception as e:
        st.error(f"Error loading album links: {e}")
        album_links_df = pd.DataFrame()
    
    # Create working copies to avoid modifying originals
    data_to_display = filtered_data.copy()
    
    # ===== ADD CACHE FORMAT HANDLING HERE =====
    # Handle cache format (Hidden Gems cache has different column names)
    if 'Source_Week' in data_to_display.columns and 'source_week' not in data_to_display.columns:
        data_to_display['source_week'] = data_to_display['Source_Week']
    
    if 'Predicted_Score' in data_to_display.columns and 'avg_score' not in data_to_display.columns:
        data_to_display['avg_score'] = data_to_display['Predicted_Score']
    
    # Ensure 'Album Name' column exists (cache has 'Album')
    if 'Album' in data_to_display.columns and 'Album Name' not in data_to_display.columns:
        data_to_display['Album Name'] = data_to_display['Album']
    # ===== END OF CACHE HANDLING =====
    
    # Keep these lines - they create copies for merging
    covers_df = album_covers_df.copy()
    links_df = album_links_df.copy()
    
    # STEP 1: Merge album covers
    try:
        # data_to_display now has BOTH 'Album' and 'Album Name' (thanks to cache handling above)
        # covers_df has: Artist, Album Name, Album Art
        
        data_to_display = data_to_display.merge(
            covers_df[['Artist', 'Album Name', 'Album Art']], 
            on=['Artist', 'Album Name'],
            how='left'
        )
    except Exception as e:
        st.error(f"Error merging album covers: {e}")
        # Continue without covers
    
    # STEP 2: Merge Spotify links
    if not links_df.empty:
        try:
            # links_df has: Album Name, Artist Name(s), Spotify URL
            # Need to match with data_to_display columns: Artist, Album
            
            # Standardize links_df column names
            if 'Artist Name(s)' in links_df.columns:
                links_df = links_df.rename(columns={'Artist Name(s)': 'Artist'})
            if 'Album Name' in links_df.columns:
                links_df = links_df.rename(columns={'Album Name': 'Album'})
            
            # Check for duplicate columns before merging
            duplicate_cols = set(links_df.columns) & set(data_to_display.columns)
            duplicate_cols.discard('Artist')
            duplicate_cols.discard('Album')
            
            if duplicate_cols:
                # Rename duplicate columns in links_df
                for col in duplicate_cols:
                    links_df = links_df.rename(columns={col: f'{col}_link'})
            
            # Merge
            data_to_display = data_to_display.merge(
                links_df[['Artist', 'Album', 'Spotify URL']],
                on=['Artist', 'Album'],
                how='left',
                suffixes=('', '_link')
            )
        except Exception as e:
            st.error(f"Error merging Spotify links: {e}")
            # Continue without links
    
    filtered_albums = data_to_display
    
    # Now display the albums using 'Artist' and 'Album' columns
    for idx, row in filtered_albums.iterrows():
        with st.container():
            st.markdown('<div class="album-container">', unsafe_allow_html=True)
            cols = st.columns([2, 4, 1, 1])
            
            with cols[0]:
                if 'Album Art' in row and pd.notna(row['Album Art']):
                    st.image(row['Album Art'], width=300, use_column_width="always")
                else:
                    st.markdown(
                        """
                        <div style="display: flex; justify-content: center; align-items: center; 
                                  height: 300px; background-color: #f0f0f0; border-radius: 10px;">
                            <span style="font-size: 48px;">🎵</span>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            with cols[1]:
                # Album title - use 'Artist' and 'Album' columns
                artist_display = row['Artist']
                album_display = row['Album']
                
                # GET ALBUM HISTORY
                album_history = get_album_history(artist_display, album_display)
                
                # Color code based on history
                if "Top 100" in album_history:
                    badge_color = "#ffd700"  # Gold
                elif "Honorable" in album_history:
                    badge_color = "#c0c0c0"  # Silver
                elif "Mid" in album_history:
                    badge_color = "#ffa500"  # Orange
                elif "Not Liked" in album_history:
                    badge_color = "#ff6b6b"  # Red
                elif "Gut Score" in album_history:
                    badge_color = "#4caf50"  # Green
                else:
                    badge_color = "#6c757d"  # Gray
                
                st.markdown(f'''
                <div class="album-title" style="font-size: 1.8rem; font-weight: 600; margin-bottom: 16px;">
                    {artist_display} - {album_display}
                    <br>
                    <span style="display: inline-block; background-color: {badge_color}; 
                           color: white; padding: 2px 8px; border-radius: 12px; 
                           font-size: 0.8rem; margin-top: 5px; font-weight: 500;">
                        {album_history}
                    </span>
                </div>
                ''', unsafe_allow_html=True)
                
                # Genres
                if 'Genres' in row and pd.notna(row['Genres']):
                    st.markdown(f'<div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;"><strong>Genre:</strong> {row["Genres"]}</div>', unsafe_allow_html=True)
                
                # Label
                if 'Label' in row and pd.notna(row['Label']):
                    st.markdown(f'<div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;"><strong>Label:</strong> {row["Label"]}</div>', unsafe_allow_html=True)
                
                # EXACT PLAYTIME
                playtime_str, track_count = calculate_exact_playtime(album_display, artist_display)
                
                if playtime_str:
                    st.markdown(f'''
                    <div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;">
                        <strong>⏱️ Playtime:</strong> {playtime_str}
                        <br>
                        <small style="color: #666;">
                        ({track_count} tracks)
                        </small>
                    </div>
                    ''', unsafe_allow_html=True)
                elif track_count > 0:
                    st.markdown(f'<div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;"><strong>Tracks:</strong> {track_count}</div>', unsafe_allow_html=True)
                elif 'Track_Count' in row:
                    st.markdown(f'<div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;"><strong>Tracks:</strong> {int(row["Track_Count"])}</div>', unsafe_allow_html=True)
                
                # SIMILAR ARTISTS WITH TOP 100 HIGHLIGHTING
                artist_for_lookup = row.get('Artist', row.get('Artist Name(s)', ''))
                if artist_for_lookup:
                    similar_artists = similar_artists_df[
                        similar_artists_df['Artist'] == artist_for_lookup
                    ]
                    
                    if not similar_artists.empty:
                        similar_list = similar_artists.iloc[0]['Similar Artists']
                        highlighted_list = highlight_top_100_in_similar(similar_list, TOP_100_ARTISTS)
                        
                        st.markdown(f'''
                        <div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;">
                            <strong>Similar Artists:</strong> {highlighted_list}
                            <br>
                            <small style="color: #666; font-style: italic;">
                            ⭐ = In your Top 100 favorites
                            </small>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Spotify link
                if 'Spotify URL' in row and pd.notna(row['Spotify URL']):
                    spotify_url = row['Spotify URL']
                    st.markdown(f'''
                    <a href="https://{spotify_url}" target="_blank" class="spotify-button">
                        ▶ Play on Spotify
                    </a>
                    ''', unsafe_allow_html=True)
            
            with cols[2]:
                # Metric container for predicted score
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                score = row.get('Predicted_Score', row.get('avg_score', 0))
                st.metric("Predicted Score", f"{score:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Gut Score section
                st.markdown('<div class="gut-score-container">', unsafe_allow_html=True)
                st.markdown("**🎯 Gut Score (0-100)**")
                
                # Get current gut score if it exists (check both column names)
                current_gut = 0
                current_notes = ""
                gut_col = None
                
                # Check which column name exists
                if 'gut_score' in row:
                    gut_col = 'gut_score'
                elif 'Gut_Score' in row:
                    gut_col = 'Gut_Score'
                    
                if gut_col and pd.notna(row[gut_col]):
                    current_gut = float(row[gut_col])
                    # Get current notes if they exist
                    if 'notes' in row and pd.notna(row['notes']):
                        current_notes = str(row['notes'])
                    # Display current gut score
                    st.markdown(f"**Current:** {int(current_gut)}")
                    if current_notes:
                        st.markdown(f"**Notes:** {current_notes}")
                else:
                    st.markdown("**Current:** Not rated")
                
                # Notes input
                notes_input = st.text_area(
                    "Notes (optional)", 
                    value=current_notes,
                    key=f"notes_{album_display}_{artist_display}",
                    height=60,
                    placeholder="Add any notes about this album..."
                )
                
                # Show current score status
                if current_gut > 0:
                    st.markdown(f"### 🎯 Current Score: {int(current_gut)}")
                    
                    # Update section
                    update_cols = st.columns([3, 2])
                    with update_cols[0]:
                        new_score = st.number_input(
                            "Update to...", 
                            0, 
                            100, 
                            int(current_gut),  # Default to current score
                            key=f"update_input_{album_display}_{artist_display}",
                            label_visibility="collapsed"
                        )
                    with update_cols[1]:
                        if st.button("✏️ Update", key=f"update_btn_{album_display}_{artist_display}"):
                            save_gut_score(album_display, artist_display, new_score, current_file_path, notes_input)
                            st.rerun()
                    
                    # Quick adjust buttons
                    st.markdown("**Quick adjust:**")
                    quick_cols = st.columns(4)
                    with quick_cols[0]:
                        if st.button("⬆️ +10", key=f"up10_{album_display}_{artist_display}"):
                            new_score = min(100, current_gut + 10)
                            save_gut_score(album_display, artist_display, new_score, current_file_path, notes_input)
                            st.rerun()
                    with quick_cols[1]:
                        if st.button("⬇️ -10", key=f"down10_{album_display}_{artist_display}"):
                            new_score = max(0, current_gut - 10)
                            save_gut_score(album_display, artist_display, new_score, current_file_path, notes_input)
                            st.rerun()
                    with quick_cols[2]:
                        if st.button("🔥 Max", key=f"max_{album_display}_{artist_display}"):
                            save_gut_score(album_display, artist_display, 100, current_file_path, notes_input)
                            st.rerun()
                    with quick_cols[3]:
                        if st.button("💀 Min", key=f"min_{album_display}_{artist_display}"):
                            save_gut_score(album_display, artist_display, 0, current_file_path, notes_input)
                            st.rerun()
                    
                else:
                    # Not scored yet
                    st.markdown("### 🎯 Rate This Album")
                    score_cols = st.columns([3, 2])
                    with score_cols[0]:
                        new_score = st.number_input(
                            "Score (0-100)", 
                            0, 
                            100, 
                            0,
                            key=f"gut_input_{album_display}_{artist_display}",
                            label_visibility="collapsed"
                        )
                    with score_cols[1]:
                        if st.button("💾 Save Score", key=f"gut_save_{album_display}_{artist_display}"):
                            save_gut_score(album_display, artist_display, new_score, current_file_path, notes_input)
                            st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with cols[3]:
                # DYNAMIC FEEDBACK DISPLAY
                try:
                    album_key = f"{artist_display}_{album_display}"
                    
                    # First check session state (for immediate feedback)
                    if album_key in st.session_state.recent_ratings:
                        gut_score = st.session_state.recent_ratings[album_key]
                        st.markdown(f'🎯 **Just rated: {gut_score}**')
                        st.balloons()  # Celebration!
                    
                    # Check MASTER gut scores file
                    master_file = 'feedback/master_gut_scores.csv'
                    
                    if os.path.exists(master_file):
                        master_df = pd.read_csv(master_file)
                        
                        # Look for this album in master file
                        album_feedback = master_df[
                            (master_df['Album'] == album_display) & 
                            (master_df['Artist'] == str(artist_display))
                        ]
                        
                        if not album_feedback.empty and pd.notna(album_feedback.iloc[0]['gut_score']):
                            gut_score = float(album_feedback.iloc[0]['gut_score'])
                            rating_date = album_feedback.iloc[0].get('gut_score_date', 'Recently')
                            
                            # Show rating with emoji based on score
                            if gut_score >= 85:
                                st.markdown(f'⭐ **{gut_score}** (Loved it!)')
                                st.caption(f"Rated on: {rating_date}")
                            elif gut_score >= 70:
                                st.markdown(f'👍 **{gut_score}** (Liked it)')
                                st.caption(f"Rated on: {rating_date}")
                            elif gut_score >= 50:
                                st.markdown(f'😐 **{gut_score}** (It was okay)')
                                st.caption(f"Rated on: {rating_date}")
                            elif gut_score >= 30:
                                st.markdown(f'👎 **{gut_score}** (Not a fan)')
                                st.caption(f"Rated on: {rating_date}")
                            else:
                                st.markdown(f'💀 **{gut_score}** (Hated it)')
                                st.caption(f"Rated on: {rating_date}")
                            
                            # Show quick comment based on score
                            comments = {
                                range(90, 101): "🔥 Absolute favorite!",
                                range(80, 90): "🎵 Really enjoyed this",
                                range(70, 80): "👂 Solid listen",
                                range(60, 70): "🤷 It was alright",
                                range(50, 60): "😕 Meh",
                                range(40, 50): "👎 Not my thing",
                                range(30, 40): "🚫 Definitely not",
                                range(0, 30): "💀 Awful"
                            }
                            
                            for score_range, comment in comments.items():
                                if int(gut_score) in score_range:
                                    st.markdown(f'<div style="font-style: italic; color: #666; margin-top: 5px;">"{comment}"</div>', unsafe_allow_html=True)
                                    break
                            
                            # Show notes if they exist
                            if 'notes' in album_feedback.columns and not pd.isna(album_feedback.iloc[0]['notes']):
                                notes = str(album_feedback.iloc[0]['notes']).strip()
                                if notes:  # Only show if not empty
                                    st.markdown(f'<div style="background-color: #f0f0f0; padding: 8px; border-radius: 4px; margin-top: 5px; font-size: 0.9rem;"><strong>📝 Notes:</strong> {notes}</div>', unsafe_allow_html=True)
                            
                            # Also check if it's in training data
                            try:
                                training_file = 'data/2026_training_complete_with_features.csv'
                                if os.path.exists(training_file):
                                    training_df = pd.read_csv(training_file)
                                    in_training = training_df[
                                        (training_df['Album Name'] == album_display) &
                                        (training_df['Artist Name(s)'].str.contains(str(artist_display), na=False)) &
                                        (training_df['source_type'] == 'gut_score_rated')
                                    ]
                                    
                                    if not in_training.empty:
                                        st.success("✅ In training data")
                                    else:
                                        st.info("⏳ Will be added Thursday")
                            except:
                                pass
                            
                        else:
                            # Not rated yet
                            if album_key not in st.session_state.recent_ratings:
                                st.markdown('😶 **Not rated yet**')
                                st.caption("Give it a listen and rate it!")
                    
                    else:
                        if album_key not in st.session_state.recent_ratings:
                            st.markdown('📝 **No ratings yet**')
                            st.caption("Be the first to rate!")
                            
                except Exception as e:
                    st.markdown('⚠️ Error loading feedback')
                    st.caption(str(e)[:50])
            
            st.markdown('</div>', unsafe_allow_html=True)
def get_album_history(artist, album):
    """Check if album appears in training data and return its history"""
    try:
        training_file = 'data/2026_training_complete_with_features.csv'
        training_df = pd.read_csv(training_file)
        
        # Find the album in training data
        album_tracks = training_df[
            (training_df['Album Name'] == album) &
            (training_df['Artist Name(s)'].str.contains(artist, na=False))
        ]
        
        if len(album_tracks) == 0:
            return "Never Rated"  # Changed from "Never rated" to "Never Rated"
        
        # Get the source type and score
        source_type = album_tracks.iloc[0]['source_type']
        score = album_tracks.iloc[0]['liked']
        
        # Map to human-readable - EXACT strings that match filters
        history_map = {
            'top_100_ranked': f"⭐ Top 100 ({score:.1f})",
            'honorable_mention': f"🏅 Honorable Mention ({score})",
            'mid': f"😐 Mid Albums ({score})",
            'not_liked': f"👎 Not Liked ({score})",
            'gut_score_rated': f"🎯 Gut Scored ({score})"
        }
        
        return history_map.get(source_type, f"Unknown: {source_type}")
        
    except Exception as e:
        return "Unknown"

def about_me_page():
    st.title("# About Me")
    st.markdown("## Hi, I'm Mike Strusz! 👋")
    st.write("""
    I'm a Data Analyst based in Milwaukee, passionate about solving real-world problems through data-driven insights. With a strong background in data analysis, visualization, and machine learning, I'm always expanding my skills to stay at the forefront of the field.  

    Before transitioning into data analytics, I spent over a decade as a teacher, where I developed a passion for making learning engaging and accessible. This experience has shaped my approach to data: breaking down complex concepts into understandable and actionable insights.  

    This project is, if I'm being honest, something I initially wanted for my own use. As an avid listener of contemporary music, I love evaluating and experiencing today's best music, often attending concerts to immerse myself in the artistry. But beyond my personal interest, this project became a fascinating exploration of how machine learning can use past behavior to predict future preferences. It's not about tracking listeners; it's about understanding patterns and applying them to create better, more personalized experiences. This approach has broad applications, from music to e-commerce to customer segmentation, and it's a powerful tool for any business looking to anticipate and meet customer needs.  
    """)
    
    st.markdown("## Let's Connect!")
    st.write("📧 Reach me at **mike.strusz@gmail.com**")
    st.write("🔗 Connect with me on [LinkedIn](https://www.linkedin.com/in/mike-strusz/) ")
    
    st.image("graphics/mike.jpeg", width=400)
    st.caption("Me on the Milwaukee Riverwalk, wearing one of my 50+ bowties.")

def album_fixer_page():
    st.title("🛠️ Album Fixer")
    st.write("Focusing on the current week's albums ONLY.")
    
    # Load ONLY the latest predictions (current week)
    latest_files = get_all_prediction_files()
    if not latest_files:
        st.error("No prediction files found.")
        return
    
    # Get the LATEST (current) week's file - first in the sorted list
    latest_file_path = latest_files[0][0]  # [0] gets first item, [0] gets file path
    predictions_df, _ = load_predictions(latest_file_path)
    
    if predictions_df is None:
        st.error("No predictions found for current week.")
        return
    
    # EXTRACT ONLY current week's albums
    # Use unique combinations of Artist and Album Name
    current_albums = predictions_df[['Artist', 'Album Name']].drop_duplicates()
    
    # Load supporting data WITHOUT cache to ensure fresh data
    album_covers_df = pd.read_csv('data/nmf_album_covers.csv') if os.path.exists('data/nmf_album_covers.csv') else pd.DataFrame(columns=['Artist', 'Album Name', 'Album Art'])
    album_links_df = pd.read_csv('data/nmf_album_links.csv') if os.path.exists('data/nmf_album_links.csv') else pd.DataFrame(columns=['Artist', 'Album Name', 'Spotify URL'])

    # STANDARDIZE COLUMN NAMES - fix for KeyError
    if 'Artist Name(s)' in album_covers_df.columns:
        album_covers_df = album_covers_df.rename(columns={'Artist Name(s)': 'Artist'})
    
    if 'Artist Name(s)' in album_links_df.columns:
        album_links_df = album_links_df.rename(columns={'Artist Name(s)': 'Artist'})
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Missing Artwork", 
        "Spotify Links", 
        "Fix Any Current Album",
        "Nuke Albums"
    ])
    
    with tab1:
        st.header("🖼️ Missing Artwork")
        missing_art = current_albums.merge(album_covers_df, on=['Artist', 'Album Name'], how='left')
        missing_art = missing_art[missing_art['Album Art'].isna()]
        
        if missing_art.empty:
            st.success("All current albums have artwork!")
        else:
            for _, row in missing_art.iterrows():
                with st.expander(f"{row['Artist']} - {row['Album Name']}"):
                    new_url = st.text_input("Album Art URL", key=f"art_{row['Artist']}_{row['Album Name']}")
                    if st.button("Update Artwork", key=f"btn_art_{row['Artist']}_{row['Album Name']}"):
                        if new_url:
                            # Update CSV
                            new_row = pd.DataFrame([{'Artist': row['Artist'], 'Album Name': row['Album Name'], 'Album Art': new_url}])
                            album_covers_df = pd.concat([album_covers_df, new_row], ignore_index=True)
                            album_covers_df.to_csv('data/nmf_album_covers.csv', index=False)
                            st.success("Artwork updated!")
                            st.rerun()

    with tab2:
        st.header("🔗 Spotify Links")
        missing_links = current_albums.merge(album_links_df, on=['Artist', 'Album Name'], how='left')
        missing_links = missing_links[missing_links['Spotify URL'].isna()]
        
        if missing_links.empty:
            st.success("All current albums have Spotify links!")
        else:
            for _, row in missing_links.iterrows():
                with st.expander(f"{row['Artist']} - {row['Album Name']}"):
                    new_link = st.text_input("Spotify URL (open.spotify.com/...)", key=f"link_{row['Artist']}_{row['Album Name']}")
                    if st.button("Update Link", key=f"btn_link_{row['Artist']}_{row['Album Name']}"):
                        if new_link:
                            # Clean link if needed
                            clean_link = new_link.replace('https://', '').replace('http://', '')
                            new_row = pd.DataFrame([{'Artist': row['Artist'], 'Album Name': row['Album Name'], 'Spotify URL': clean_link}])
                            album_links_df = pd.concat([album_links_df, new_row], ignore_index=True)
                            album_links_df.to_csv('data/nmf_album_links.csv', index=False)
                            st.success("Link updated!")
                            st.rerun()

    with tab3:
        st.header("🔧 Fix Any Current Album")
        selected_album = st.selectbox(
            "Select an album to fix",
            options=[f"{r['Artist']} - {r['Album Name']}" for _, r in current_albums.iterrows()]
        )
        
        if selected_album:
            artist = selected_album.split(' - ')[0]
            album = selected_album.split(' - ')[1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Current Artwork")
                current_art = album_covers_df[(album_covers_df['Artist'] == artist) & (album_covers_df['Album Name'] == album)]
                if not current_art.empty:
                    st.image(current_art.iloc[0]['Album Art'], width=200)
                new_art = st.text_input("New Artwork URL")
                if st.button("Update Art"):
                    # Remove old and add new
                    album_covers_df = album_covers_df[~((album_covers_df['Artist'] == artist) & (album_covers_df['Album Name'] == album))]
                    new_row = pd.DataFrame([{'Artist': artist, 'Album Name': album, 'Album Art': new_art}])
                    album_covers_df = pd.concat([album_covers_df, new_row], ignore_index=True)
                    album_covers_df.to_csv('data/nmf_album_covers.csv', index=False)
                    st.success("Art updated!")
                    st.rerun()
            
            with col2:
                st.subheader("Current Link")
                current_link = album_links_df[(album_links_df['Artist'] == artist) & (album_links_df['Album Name'] == album)]
                if not current_link.empty:
                    st.write(current_link.iloc[0]['Spotify URL'])
                new_link = st.text_input("New Spotify URL")
                if st.button("Update Spotify Link"):
                    clean_link = new_link.replace('https://', '').replace('http://', '')
                    album_links_df = album_links_df[~((album_links_df['Artist'] == artist) & (album_links_df['Album Name'] == album))]
                    new_row = pd.DataFrame([{'Artist': artist, 'Album Name': album, 'Spotify URL': clean_link}])
                    album_links_df = pd.concat([album_links_df, new_row], ignore_index=True)
                    album_links_df.to_csv('data/nmf_album_links.csv', index=False)
                    st.success("Link updated!")
                    st.rerun()

    with tab4:
        st.header("☢️ Nuke Albums")
        st.write("Remove albums that shouldn't be in the list (e.g., singles, re-releases).")
        
        nuked_df = load_nuked_albums()
        
        with st.form("nuke_form"):
            nuke_artist = st.selectbox("Artist", options=sorted(current_albums['Artist'].unique()))
            nuke_album = st.selectbox("Album", options=sorted(current_albums[current_albums['Artist'] == nuke_artist]['Album Name'].unique()))
            nuke_reason = st.text_input("Reason (e.g., Single, Re-release, Wrong Genre)")
            submit = st.form_submit_button("Nuke Album")
            
            if submit:
                new_nuke = pd.DataFrame([{'Artist': nuke_artist, 'Album Name': nuke_album, 'Reason': nuke_reason}])
                nuked_df = pd.concat([nuked_df, new_nuke], ignore_index=True)
                nuked_df.to_csv('data/nuked_albums.csv', index=False)
                st.success(f"Nuked {nuke_album}!")
                st.rerun()
        
        if not nuked_df.empty:
            st.subheader("Currently Nuked")
            st.dataframe(nuked_df)

def notebook_page():
    st.title("📓 The Machine Learning Model in my Jupyter Notebook")
    st.subheader("Embedded notebook content below:")
    
    try:
        with open('graphics/Music_Taste_Machine_Learning_Data_Prep.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        st.markdown('<div class="notebook-content">', unsafe_allow_html=True)
        components.html(html_content, height=800, scrolling=True)
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading notebook content: {e}")

def historical_rating_page():
    st.title("📅 Rate Historical Albums")
    
    # --- LOAD REQUIRED DATA ---
    album_covers_df = load_album_covers()
    similar_artists_df = load_similar_artists()
    
    # Get all prediction files
    prediction_files = []
    for file in glob.glob('predictions/*_Album_Recommendations.csv'):
        try:
            date_str = os.path.basename(file).split('_')[0]
            date_obj = datetime.strptime(date_str, '%m-%d-%y')
            readable_date = date_obj.strftime('%B %d, %Y')
            prediction_files.append((readable_date, file, date_obj))
        except:
            continue
    
    # Sort by date (newest first)
    prediction_files.sort(key=lambda x: x[2], reverse=True)
    
    # === MODE SELECTOR ===
    st.subheader("🎛️ Exploration Mode")
    mode = st.radio(
        "Choose how to explore:",
        ["🎲 Random Album Explorer", "📅 Browse by Month", "📆 Single Week"],
        horizontal=True
    )
    
    if mode == "🎲 Random Album Explorer":
        # ========== RANDOM ALBUM EXPLORER ==========
        st.subheader("🎲 Random Album Explorer")
        st.write("Discover random albums from any time period!")
        
        # === ONE-CLICK INSTANT EXPLORATION ===
        st.subheader("🎲 One-Click Instant Exploration")
        st.write("Click a button, get 3 random albums INSTANTLY!")
        
        btn_col1, btn_col2 = st.columns(2)  # Only 2 buttons - Hidden Gems and Random Mix
        
        with btn_col1:
            hidden_gems = st.button(
                "**🎯\nHidden Gems**\n*210 pre-computed*\n*Never rated + predicted ≥75*", 
                use_container_width=True, 
                type="primary",
                key="hidden_gems_btn"
            )
                
        with btn_col2:
            random_mix = st.button(
                "**🎪\nRandom Mix**\n*Anything goes!*\n*From recent months*", 
                use_container_width=True, 
                type="secondary",
                key="random_mix_btn"
            )
        
        # HIDDEN GEMS - INSTANT FROM CACHE
        if hidden_gems:
            # Load the pre-computed Hidden Gems sample
            hidden_gems_df = load_hidden_gems_sample()
            
            if hidden_gems_df.empty:
                st.error("No Hidden Gems found! Run update_hidden_gems.py first.")
                return
            
            # Get 3 random gems from the sample
            if len(hidden_gems_df) >= 3:
                random_albums = hidden_gems_df.sample(n=3, random_state=None)  # Different each time
            else:
                random_albums = hidden_gems_df
            
            st.success(f"🎉 Found {len(random_albums)} Hidden Gems for you! (from 210 total)")
            
            # Show sources
            if 'Source_Week' in random_albums.columns and len(random_albums['Source_Week'].unique()) > 0:
                weeks = random_albums['Source_Week'].unique()[:2]
                week_str = ", ".join(weeks)
                if len(random_albums['Source_Week'].unique()) > 2:
                    week_str += f" and {len(random_albums['Source_Week'].unique()) - 2} more"
                st.caption(f"From weeks: {week_str}")
            
            # Rename columns to match what display_album_predictions expects
            if 'Album' in random_albums.columns and 'Album Name' not in random_albums.columns:
                random_albums['Album Name'] = random_albums['Album']
            if 'Predicted_Score' in random_albums.columns and 'avg_score' not in random_albums.columns:
                random_albums['avg_score'] = random_albums['Predicted_Score']
            
            display_album_predictions(random_albums, album_covers_df, similar_artists_df, "hidden_gems")
            
        # RANDOM MIX - keep the old logic (or simplify it)
        elif random_mix:
            # Simple random mix from recent months
            sample_from = ["Never Rated", "Mid Albums", "Honorable Mention", "Top 100", "Not Liked", "Gut Scored"]
            min_score = 0
            num_albums = 3
            button_name = "Random Mix"
            
            # Use recent albums only (last 3 months for speed)
            max_date = max(date_obj for _, _, date_obj in prediction_files)
            cutoff_date = max_date - timedelta(days=30*3)
            
            # Execute the search
            with st.spinner(f"Finding {num_albums} random albums..."):
                # Load recent albums
                all_albums_list = []
                for date, file, date_obj in prediction_files:
                    if date_obj < cutoff_date:
                        continue
                    
                    df, _ = load_predictions(file)
                    if len(df) > 0:
                        df['source_week'] = date
                        df['source_date'] = date_obj
                        all_albums_list.append(df)
                
                if not all_albums_list:
                    st.error("No albums found in the last 3 months!")
                    return
                
                combined = pd.concat(all_albums_list, ignore_index=True)
                
                # Filter by type
                filtered_indices = []
                for idx, row in combined.iterrows():
                    history = get_album_history(row['Artist'], row['Album'])
                    if any(rating_type in history for rating_type in sample_from):
                        filtered_indices.append(idx)
                
                # Check if we have enough
                if len(filtered_indices) < num_albums:
                    st.warning(f"Only found {len(filtered_indices)} albums")
                    num_albums = min(num_albums, len(filtered_indices))
                
                if num_albums > 0:
                    # Random sample
                    random_indices = random.sample(filtered_indices, num_albums)
                    random_albums = combined.loc[random_indices]
                    
                    st.success(f"🎉 Found {len(random_albums)} random albums for you!")
                    
                    # Show sources
                    if len(random_albums['source_week'].unique()) > 0:
                        weeks = random_albums['source_week'].unique()[:2]
                        week_str = ", ".join(weeks)
                        if len(random_albums['source_week'].unique()) > 2:
                            week_str += f" and {len(random_albums['source_week'].unique()) - 2} more"
                        st.caption(f"From weeks: {week_str}")
                    
                    display_album_predictions(random_albums, album_covers_df, similar_artists_df, "random_mix")
                else:
                    st.info(f"No random albums found. Try Hidden Gems instead!")
        
        else:
            # No button clicked yet, show stats
            hidden_gems_df = load_hidden_gems_cache()
            total_gems = len(hidden_gems_df) if not hidden_gems_df.empty else 0
            
            st.info(f"""
            👆 **Click a button to explore!**
            
            **Hidden Gems Cache Status:**
            - ✅ {total_gems} pre-computed Hidden Gems
            - 🎯 Never rated + predicted score ≥75
            - ⚡ Instant results from cache
            """)

def dacus_game_page(G):
    st.title("🎵 6 Degrees of Lucy Dacus")
    st.write("""
    ### How It Works
    Select an artist to see how closely they're connected to Lucy Dacus!
    The **Dacus number** is the number of connections between the artist and Lucy Dacus.
    """)
    
    # Get all artists from the graph
    all_artists = sorted(list(G.nodes()))
    
    # Create a search box with autocomplete
    search_term = st.text_input("Search for an artist:", "")
    
    # Filter artists based on search term (case-insensitive)
    if search_term:
        filtered_artists = [artist for artist in all_artists 
                          if search_term.lower() in artist.lower()]
        
        # Display "no results" message if needed
        if not filtered_artists:
            st.warning(f"No artists found matching '{search_term}'")
            return
            
        # Limit the number of suggestions to prevent overwhelming the UI
        if len(filtered_artists) > 10:
            st.info(f"Found {len(filtered_artists)} matches. Showing top 10.")
            filtered_artists = filtered_artists[:10]
            
        # Let user select from filtered results
        selected_artist = st.selectbox(
            "Select an artist:", 
            options=filtered_artists
        )
    else:
        # If no search term, show popular artists or a subset
        popular_artists = ["Phoebe Bridgers", "Boygenius", "Julien Baker", 
                          "Japanese Breakfast", "Mitski", "Big Thief", 
                          "The National", "Snail Mail", "Soccer Mommy"]
        # Ensure these artists are in the graph
        popular_artists = [a for a in popular_artists if a in G.nodes()]
        
        st.write("Or select from popular artists:")
        selected_artist = st.selectbox(
            "Popular artists:", 
            options=popular_artists + ["Select an artist..."],
            index=len(popular_artists)  # Default to "Select an artist..."
        )
        
        if selected_artist == "Select an artist...":
            st.info("Please search for an artist or select one from the list")
            return
    
    # Calculate Dacus number and path
    dacus_number, path = calculate_dacus_number(selected_artist, G)
    
    if dacus_number is not None:
        st.success(f"**Dacus Number:** {dacus_number}")
        st.write(f"**Path to Lucy Dacus:** {' → '.join(path)}")
        
        # Visualize the path
        st.subheader("Network Path Visualization")
        with st.spinner("Generating network visualization..."):
            # Create a subgraph with the path and some neighbors for context
            path_nodes = set(path)
            context_nodes = set()
            
            # Add some context nodes (neighbors of path nodes)
            for node in path:
                neighbors = list(G.neighbors(node))[:3]  # Limit to 3 neighbors
                context_nodes.update(neighbors)
            
            all_viz_nodes = path_nodes.union(context_nodes)
            subgraph = G.subgraph(all_viz_nodes)
            
            # Create and display the visualization
            fig = visualize_artist_network(subgraph, path)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No path found. This artist might not be connected to Lucy Dacus in our network.")
        
def calculate_dacus_number(artist_name, G):
    """
    Calculate the Dacus number and path for a given artist.
    """
    try:
        if artist_name not in G:
            return None, None
        
        if artist_name == "Lucy Dacus":
            return 0, ["Lucy Dacus"]
        
        path = nx.shortest_path(G, source=artist_name, target="Lucy Dacus")
        dacus_number = len(path) - 1
        return dacus_number, path
    except nx.NetworkXNoPath:
        return None, None

def visualize_artist_network(G, path):
    """
    Visualize the artist network and highlight the path to Lucy Dacus.
    """
    pos = nx.spring_layout(G, seed=42)
    
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))
    
    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers+text', hoverinfo='text',
        marker=dict(size=10, color='lightblue'),
        textposition="top center"
    )
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (node,)
    
    # Highlight the path
    path_edges = list(zip(path[:-1], path[1:]))
    path_trace = []
    for edge in path_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        path_trace.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=2, color='red'),
            hoverinfo='none',
            mode='lines'
        ))
    
    fig = go.Figure(data=edge_trace + [node_trace] + path_trace)
    fig.update_layout(showlegend=False, hovermode='closest')
    return fig

def build_graph(df, df_liked_similar, include_nmf=False):
    """
    Build a graph of artists and their connections.
    Only includes liked artists and their similar artists by default.
    Optionally includes NMF and not-liked artists (without adding edges).
    """
    G = nx.Graph()
    
    # Add nodes for liked artists
    if 'playlist_origin' in df.columns and 'Artist' in df.columns:
        liked_artists = set(
            df[df['playlist_origin'].isin(['df_liked', 'df_fav_albums'])]['Artist']
            .str.split(',').explode().str.strip()
        )
    else:
        liked_artists = set()  # Fallback if columns are missing
    
    G.add_nodes_from(liked_artists, type='liked')
    
    # Add nodes for similar artists (from liked)
    if 'Similar Artists' in df_liked_similar.columns:
        similar_artists_liked = set(
            df_liked_similar['Similar Artists']
            .dropna()
            .str.split(',').explode().str.strip()
        )
    else:
        similar_artists_liked = set()  # Fallback if column is missing
    
    G.add_nodes_from(similar_artists_liked, type='similar_liked')
    
    # Add edges based on similarity (from liked)
    if 'Artist' in df_liked_similar.columns and 'Similar Artists' in df_liked_similar.columns:
        for _, row in df_liked_similar.iterrows():
            artist = row['Artist']
            if isinstance(row['Similar Artists'], str):
                similar = row['Similar Artists'].split(', ')
                for s in similar:
                    G.add_edge(artist, s, weight=1.0)
    
    # Optionally include NMF and not-liked artists (without adding edges)
    if include_nmf and 'playlist_origin' in df.columns and 'Artist' in df.columns:
        nmf_artists = set(
            df[df['playlist_origin'] == 'df_nmf']['Artist']
            .str.split(',').explode().str.strip()
        )
        not_liked_artists = set(
            df[df['playlist_origin'] == 'df_not_liked']['Artist']
            .str.split(',').explode().str.strip()
        )
        G.add_nodes_from(nmf_artists, type='nmf')
        G.add_nodes_from(not_liked_artists, type='not_liked')
    
    return G

def main():
    # ORIGINAL SIDEBAR RESTORED
    st.sidebar.title("About This Project")
    st.sidebar.write("""
    ### Tech Stack
    - 🤖 Machine Learning: RandomForest & XGBoost
    - 📊 Data Processing: Pandas & NumPy
    - 🎨 Visualization: Plotly & Streamlit
    - 🎵 Data Source: Spotify & Lastfm APIs
    
    ### Key Features
    - Weekly New Music Predictions
    - Advanced Artist Similarity Analysis
    - Genre-based Learning
    - Automated Label Analysis
    """)
    
    # Add the "Clear Cache and Refresh Data" button
    if st.sidebar.button("Clear Cache and Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # ===== HIDDEN GEMS CACHE SECTION =====
    st.sidebar.markdown("---")
    st.sidebar.header("🔄 Hidden Gems Cache")
    
    # Show cache stats
    cache_file = 'data/hidden_gems_cache.csv'
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            st.sidebar.metric("Hidden Gems in Cache", len(df))
            avg_score = df['Predicted_Score'].mean() if 'Predicted_Score' in df.columns else "N/A"
            st.sidebar.metric("Avg Score", f"{avg_score:.1f}" if isinstance(avg_score, (int, float)) else avg_score)
        except Exception as e:
            st.sidebar.error(f"Error reading cache: {e}")
    else:
        st.sidebar.info("No cache found. Run update script.")
    
    if st.sidebar.button("🔄 Update Hidden Gems Cache"):
        with st.spinner("Updating Hidden Gems cache..."):
            try:
                import subprocess
                result = subprocess.run(
                    ["python", "update_hidden_gems.py"], 
                    capture_output=True, 
                    text=True,
                    cwd=os.getcwd()
                )
                
                output_text = ""
                if result.stdout:
                    output_text += result.stdout
                if result.stderr:
                    output_text += f"\nErrors:\n{result.stderr}"
                
                if output_text:
                    st.sidebar.text_area("Update Output:", output_text, height=200)
                
                if result.returncode == 0:
                    st.sidebar.success("✅ Cache updated!")
                else:
                    st.sidebar.error("❌ Update failed")
                
                # Clear cache to force reload
                st.cache_data.clear()
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
    # ===== END OF HIDDEN GEMS SECTION =====
    
    # Navigation - ORIGINAL OPTIONS RESTORED
    st.sidebar.markdown("---")
    st.sidebar.header("📱 Navigation")
    page_options = [
        "Weekly Predictions",
        "Historical Ratings",
        "The Machine Learning Model",
        "6 Degrees of Lucy Dacus",
        "About Me"
    ]

    # Add Album Fixer only if running locally
    if IS_LOCAL:
        page_options.append("Album Fixer")

    page = st.sidebar.radio("Navigate", page_options)
    
    # Load data needed for multiple pages
    album_covers_df = load_album_covers()
    similar_artists_df = load_similar_artists()
    liked_similar_df = load_similar_artists()  # Use the same function
    
    if page == "Weekly Predictions":
        st.title("🎵 New Music Friday Regression Model")
        st.subheader("Personalized New Music Friday Recommendations")
        
            # ===== TOP FILTERS SECTION =====
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"**📅 Most Recent:** {selected_date_str}")
                
            with col2:
                # Genre Filter
                if 'Genres' in predictions_df.columns:
                    # Extract all unique genres
                    all_genres = set()
                    for genres in predictions_df['Genres'].dropna():
                        for g in str(genres).split(','):
                            all_genres.add(g.strip())
                    
                    sorted_genres = sorted(list(all_genres))
                    selected_genres = st.multiselect("🎵 Filter by Genre", options=sorted_genres)
                    
                    if selected_genres:
                        # Filter rows that contain ANY of the selected genres
                        predictions_df = predictions_df[
                            predictions_df['Genres'].apply(lambda x: any(g.strip() in selected_genres for g in str(x).split(',')) if pd.notna(x) else False)
                        ]

            with col3:
                # Search Filter
                search_query = st.text_input("🔍 Search Artist/Album", "").lower()
                
            with col4:
                # Sort Order
                sort_order = st.selectbox("📊 Sort by", ["Highest First", "Lowest First"])
            
            st.markdown("---")
            # ===== END TOP FILTERS =====
            # Apply search
            if search_query:
                predictions_df = predictions_df[
                    predictions_df['Artist'].str.lower().str.contains(search_query) |
                    predictions_df['Album Name'].str.lower().str.contains(search_query)
                ]
            
            # Apply sort
            score_col = 'Predicted_Score' if 'Predicted_Score' in predictions_df.columns else 'avg_score'
            predictions_df = predictions_df.sort_values(
                by=score_col, 
                ascending=(sort_order == "Lowest First")
            )
            
            # Genre Filter
            if 'Genres' in predictions_df.columns:
                # Extract all unique genres
                all_genres = set()
                for genres in predictions_df['Genres'].dropna():
                    for g in str(genres).split(','):
                        all_genres.add(g.strip())
                
                sorted_genres = sorted(list(all_genres))
                selected_genres = st.sidebar.multiselect("Filter by Genre", options=sorted_genres)
                
                if selected_genres:
                    # Filter rows that contain ANY of the selected genres
                    predictions_df = predictions_df[
                        predictions_df['Genres'].apply(lambda x: any(g.strip() in selected_genres for g in str(x).split(',')) if pd.notna(x) else False)
                    ]

            # Sort Order
            sort_order = st.sidebar.selectbox("Sort by Score", ["Highest First", "Lowest First"])
            
            # Apply search
            if search_query:
                predictions_df = predictions_df[
                    predictions_df['Artist'].str.lower().str.contains(search_query) |
                    predictions_df['Album Name'].str.lower().str.contains(search_query)
                ]
            
            # Apply sort
            score_col = 'Predicted_Score' if 'Predicted_Score' in predictions_df.columns else 'avg_score'
            predictions_df = predictions_df.sort_values(
                by=score_col, 
                ascending=(sort_order == "Lowest First")
            )
            
            st.write(f"Showing {len(predictions_df)} albums for {analysis_date}")
            display_album_predictions(predictions_df, album_covers_df, similar_artists_df, selected_file)
            
    elif page == "Historical Ratings":  # <-- ADD THIS NEW BLOCK
        historical_rating_page()
            
    elif page == "Album Fixer":
        album_fixer_page()
        
    elif page == "The Machine Learning Model":
        notebook_page()
        
    elif page == "6 Degrees of Lucy Dacus":
        # Load the latest predictions for the graph
        latest_file = get_all_prediction_files()[0][0] if get_all_prediction_files() else None
        predictions_data = load_predictions(latest_file)
        
        if predictions_data is None:
            st.error("Could not load prediction data. Please check the predictions folder.")
            return
            
        df, _ = predictions_data
        
        # Build the artist network graph
        G = build_graph(df, liked_similar_df, include_nmf=True)
        dacus_game_page(G)
        
    elif page == "About Me":
        about_me_page()

if __name__ == "__main__":
    main()