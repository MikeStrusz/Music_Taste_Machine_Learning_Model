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
        album_tracks = pd.DataFrame()

        if os.path.exists(training_file):
            training_df = pd.read_csv(training_file)
            album_tracks = training_df[
                (training_df['Album Name'] == album_name) &
                (training_df['Artist Name(s)'].str.contains(artist, na=False, case=False))
            ]

        # Fallback: check NMF archives for current week albums
        if len(album_tracks) == 0:
            archive_files = sorted(
                glob.glob('data/archived_nmf_with_features/*.csv'), reverse=True
            )
            for archive_file in archive_files:
                try:
                    archive_df = pd.read_csv(archive_file)
                    hits = archive_df[
                        (archive_df['Album Name'] == album_name) &
                        (archive_df['Artist Name(s)'].str.contains(artist, na=False, case=False))
                    ]
                    if len(hits) > 0:
                        album_tracks = hits
                        break
                except Exception:
                    continue

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

@st.cache_data
def load_training_history():
    """Load training data and master gut scores once, cache forever"""
    training_file = 'data/2026_training_complete_with_features.csv'
    training_df = pd.read_csv(training_file) if os.path.exists(training_file) else pd.DataFrame()
    master_file = 'feedback/master_gut_scores.csv'
    master_df = pd.read_csv(master_file) if os.path.exists(master_file) else pd.DataFrame()
    return training_df, master_df

def highlight_top_100_in_similar(similar_artists_str, top_100_artists):
    """
    Return the list of similar artists as plain text (no bolding or stars).
    """
    if pd.isna(similar_artists_str) or not similar_artists_str:
        return "No similar artists data"
    
    similar_list = [artist.strip() for artist in similar_artists_str.split(',')]
    highlighted = []
    
    for artist in similar_list:
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

@st.cache_data(ttl=1)
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

import hashlib

def cover_filename(artist, album):
    """Deterministic local cover filename — must match notebook version"""
    key = f"{str(artist).strip().lower()}_{str(album).strip().lower()}"
    return os.path.join('covers', hashlib.md5(key.encode()).hexdigest() + '.jpg')

@st.cache_data(ttl=1)
def load_album_covers():
    try:
        df = pd.read_csv('data/nmf_album_covers.csv')
        if 'Album Name' in df.columns and 'Album' not in df.columns:
            df = df.rename(columns={'Album Name': 'Album'})
        if 'Artist Name(s)' in df.columns and 'Artist' not in df.columns:
            df = df.rename(columns={'Artist Name(s)': 'Artist'})
        df = df.drop_duplicates(subset=['Artist', 'Album'], keep='first')
        # Prefer local file over remote URL
        df['Album Art'] = df.apply(
            lambda row: cover_filename(row['Artist'], row['Album'])
            if os.path.exists(cover_filename(row['Artist'], row['Album']))
            else row['Album Art'],
            axis=1
        )
        return df
    except Exception as e:
        st.error(f"Error loading album covers data: {e}")
        return pd.DataFrame(columns=['Artist', 'Album', 'Album Art'])

@st.cache_data(ttl=1)
def load_album_links():
    frames = []

    # Source 1: nmf_album_links.csv
    try:
        df = pd.read_csv('data/nmf_album_links.csv')
        if 'Artist Name(s)' in df.columns and 'Artist' not in df.columns:
            df = df.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in df.columns and 'Album' not in df.columns:
            df = df.rename(columns={'Album Name': 'Album'})
        frames.append(df[['Artist', 'Album', 'Spotify URL']])
    except Exception:
        pass

    # Source 2: album_metadata_cache.csv
    try:
        df2 = pd.read_csv('data/album_metadata_cache.csv')
        df2 = df2.rename(columns={'artist': 'Artist', 'album': 'Album', 'spotify_url': 'Spotify URL'})
        frames.append(df2[['Artist', 'Album', 'Spotify URL']])
    except Exception:
        pass

    if not frames:
        return pd.DataFrame(columns=['Artist', 'Album', 'Spotify URL'])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['Artist', 'Album'], keep='last')
    return combined

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

            # Normalize column names to match notebook output
            if 'Album' in predictions_df.columns and 'Album Name' not in predictions_df.columns:
                predictions_df['Album Name'] = predictions_df['Album']
            if 'Artist Name(s)' in predictions_df.columns and 'Artist' not in predictions_df.columns:
                predictions_df['Artist'] = predictions_df['Artist Name(s)']

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
        
        # Clear cache so predictions reload with new gut score
        st.cache_data.clear()
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
    
    # STEP 1: Merge album covers — both normalized to 'Album' column
    try:
        data_to_display = data_to_display.merge(
            covers_df[['Artist', 'Album', 'Album Art']],
            on=['Artist', 'Album'],
            how='left'
        )
    except Exception as e:
        st.error(f"Error merging album covers: {e}")

    # STEP 2: Merge Spotify links — both normalized to 'Album' column
    if not links_df.empty:
        try:
            data_to_display = data_to_display.merge(
                links_df[['Artist', 'Album', 'Spotify URL']],
                on=['Artist', 'Album'],
                how='left'
            )
        except Exception as e:
            st.error(f"Error merging Spotify links: {e}")
    
    filtered_albums = data_to_display
    
    # Now display the albums using 'Artist' and 'Album' columns
    for idx, row in filtered_albums.iterrows():
        with st.container():
            st.markdown('<div class="album-container">', unsafe_allow_html=True)
            cols = st.columns([2, 4, 2])
            
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
                    # Fix cover UI — only shows when cover is missing
                    fix_key = f"fix_cover_{row['Artist']}_{row['Album']}"
                    if not st.session_state.get(f"cover_saved_{fix_key}"):
                        with st.expander("🖼️ Fix Cover", expanded=False):
                            search_query = f"{row['Artist']} {row['Album']} album cover"
                            google_url = f"https://www.google.com/search?tbm=isch&q={search_query.replace(' ', '+')}"
                            st.markdown(f"[🔍 Search Google Images]({google_url})", unsafe_allow_html=True)
                            new_url = st.text_input("Paste image URL", key=f"cover_url_{fix_key}")
                            if st.button("💾 Save Cover", key=f"cover_save_{fix_key}"):
                                if new_url:
                                    import hashlib, requests as req
                                    key = f"{str(row['Artist']).strip().lower()}_{str(row['Album']).strip().lower()}"
                                    local_path = os.path.join('covers', hashlib.md5(key.encode()).hexdigest() + '.jpg')
                                    try:
                                        os.makedirs('covers', exist_ok=True)
                                        img_response = req.get(new_url, timeout=10)
                                        img_response.raise_for_status()
                                        with open(local_path, 'wb') as f:
                                            f.write(img_response.content)
                                        covers_csv = pd.read_csv('data/nmf_album_covers.csv')
                                        new_row = pd.DataFrame([{'Artist': row['Artist'], 'Album Name': row['Album'], 'Album Art': local_path}])
                                        covers_csv = pd.concat([covers_csv, new_row], ignore_index=True)
                                        covers_csv = covers_csv.drop_duplicates(subset=['Artist', 'Album Name'], keep='last')
                                        covers_csv.to_csv('data/nmf_album_covers.csv', index=False)
                                        st.session_state[f"cover_saved_{fix_key}"] = True
                                        st.cache_data.clear()
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Failed: {e}")
            
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

                # Define current_gut early so title block can use it
                current_gut = 0
                current_notes = ""
                gut_col = None
                if 'gut_score' in row:
                    gut_col = 'gut_score'
                elif 'Gut_Score' in row:
                    gut_col = 'Gut_Score'
                if gut_col and pd.notna(row[gut_col]):
                    current_gut = float(row[gut_col])
                    if 'notes' in row and pd.notna(row['notes']):
                        current_notes = str(row['notes'])
                
                if current_gut > 0:
                    review_html = f"<div style='font-style: italic; color: #888; font-size: 0.95rem; margin-top: 4px;'>{current_notes}</div>" if current_notes else ""
                    st.markdown(f'''
                    <div class="album-title" style="font-size: 1.8rem; font-weight: 600; margin-bottom: 4px;">
                        {artist_display} - {album_display}
                        <br>
                        <span style="font-size: 2rem; font-weight: 700; color: #4caf50;">{int(current_gut)}</span>
                        {review_html}
                    </div>
                    ''', unsafe_allow_html=True)
                else:
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

                # Similar Artists
                artist_for_lookup = row.get('Artist', row.get('Artist Name(s)', ''))
                if artist_for_lookup:
                    similar_artists = similar_artists_df[
                        similar_artists_df['Artist'] == artist_for_lookup
                    ]
                    if not similar_artists.empty:
                        similar_list = similar_artists.iloc[0]['Similar Artists']
                        highlighted_list = highlight_top_100_in_similar(similar_list, TOP_100_ARTISTS)
                        has_top_100 = '⭐' in highlighted_list
                        st.markdown(f'''
                        <div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;">
                            <strong>Similar Artists:</strong> {highlighted_list}
                            {"<br><small style='color: #666; font-style: italic;'>⭐ = In your Top 100 favorites</small>" if has_top_100 else ""}
                        </div>
                        ''', unsafe_allow_html=True)

                # Playtime
                playtime_str, track_count = calculate_exact_playtime(album_display, artist_display)
                if playtime_str:
                    st.markdown(f'<div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;"><strong>⏱️ Playtime:</strong> {playtime_str}</div>', unsafe_allow_html=True)
                elif track_count > 0:
                    st.markdown(f'<div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;"><strong>Tracks:</strong> {track_count}</div>', unsafe_allow_html=True)
                elif 'Track_Count' in row:
                    st.markdown(f'<div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;"><strong>Tracks:</strong> {int(row["Track_Count"])}</div>', unsafe_allow_html=True)

                # Label
                if 'Label' in row and pd.notna(row['Label']):
                    st.markdown(f'<div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;"><strong>Label:</strong> {row["Label"]}</div>', unsafe_allow_html=True)

                # Release Date
                release_date = None
                if 'Release Date' in row and pd.notna(row['Release Date']):
                    release_date = str(row['Release Date'])
                if not release_date and current_file_path and current_file_path not in ['hidden_gems', 'random_mix', 'random_draw']:
                    try:
                        date_str = os.path.basename(current_file_path).split('_')[0]
                        release_date = datetime.strptime(date_str, '%m-%d-%y').strftime('%B %d, %Y')
                    except:
                        pass
                if release_date:
                    try:
                        release_date = datetime.strptime(release_date, '%Y-%m-%d').strftime('%B %d, %Y')
                    except:
                        pass
                    st.markdown(f'<div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;"><strong>Released:</strong> {release_date}</div>', unsafe_allow_html=True)

                # Spotify link
                if 'Spotify URL' in row and pd.notna(row['Spotify URL']):
                    spotify_url = str(row['Spotify URL']).strip()
                    if not spotify_url.startswith('http'):
                        spotify_url = 'https://' + spotify_url
                    st.link_button("▶ Play on Spotify", spotify_url)
            
            with cols[2]:
                score = row.get('Predicted_Score', row.get('avg_score', 0))
                st.metric("Predicted Score", f"{score:.1f}")
                
                # Gut Score section
                st.markdown("<div style='font-family: Lora, serif; font-style: italic; font-size: 1.6rem; color: #333; margin-bottom: 8px;'>Rate This Album</div>", unsafe_allow_html=True)
                file_key = os.path.basename(current_file_path) if current_file_path else "default"
                new_score = st.number_input(
                    "Score (0-100)",
                    0,
                    100,
                    int(current_gut) if current_gut > 0 else None,
                    key=f"gut_input_{file_key}_{album_display}_{artist_display}",
                    label_visibility="collapsed"
                )
                notes_input = st.text_area(
                    "Notes (optional)",
                    value=current_notes,
                    key=f"notes_{file_key}_{album_display}_{artist_display}",
                    height=60,
                    placeholder="Add any notes about this album..."
                )
                if st.button("💾 Save Score", key=f"gut_save_{file_key}_{album_display}_{artist_display}"):
                    save_gut_score(album_display, artist_display, new_score, current_file_path, notes_input)
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)
@st.cache_data
def get_album_history(artist, album):
    """Check if album appears in training data and return its history"""
    try:
        training_df, master_df = load_training_history()

        album_tracks = training_df[
            (training_df['Album Name'] == album) &
            (training_df['Artist Name(s)'].str.contains(artist, na=False))
        ]

        if len(album_tracks) == 0:
            if not master_df.empty:
                gut_match = master_df[
                    (master_df['Album'] == album) &
                    (master_df['Artist'] == artist) &
                    (master_df['gut_score'].notna())
                ]
                if not gut_match.empty:
                    score = gut_match.iloc[0]['gut_score']
                    return f"🎯 Gut Scored ({score})"
            return "Never Rated"

        source_type = album_tracks.iloc[0]['source_type']
        score = album_tracks.iloc[0]['liked']

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
    st.cache_data.clear()

    # STANDARDIZE COLUMN NAMES - fix for KeyError
    if 'Artist Name(s)' in album_covers_df.columns:
        album_covers_df = album_covers_df.rename(columns={'Artist Name(s)': 'Artist'})
    
    if 'Artist Name(s)' in album_links_df.columns:
        album_links_df = album_links_df.rename(columns={'Artist Name(s)': 'Artist'})
    
    tab1, tab2 = st.tabs([
        "✏️ Edit Album",
        "☢️ Nuke Albums",
    ])

    with tab1:
        st.header("✏️ Edit Album")

        # Filter buttons to surface problem albums
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        with filter_col1:
            if st.button("🖼️ Missing Cover", key="filter_cover", use_container_width=True):
                st.session_state['fixer_filter'] = 'cover'
        with filter_col2:
            if st.button("🔗 Missing Link", key="filter_link", use_container_width=True):
                st.session_state['fixer_filter'] = 'link'
        with filter_col3:
            if st.button("🎵 Missing Genre", key="filter_genre", use_container_width=True):
                st.session_state['fixer_filter'] = 'genre'
        with filter_col4:
            if st.button("✕ Clear Filter", key="filter_clear", use_container_width=True):
                st.session_state['fixer_filter'] = None

        # Build filtered album list
        fixer_filter = st.session_state.get('fixer_filter')
        album_options = []
        for _, r in current_albums.iterrows():
            artist = r['Artist']
            album = r['Album Name']
            if fixer_filter == 'cover':
                has_cover = not album_covers_df[
                    (album_covers_df['Artist'] == artist) & (album_covers_df['Album Name'] == album)
                ].empty
                if has_cover:
                    continue
            elif fixer_filter == 'link':
                has_link = not album_links_df[
                    (album_links_df['Artist'] == artist) & (album_links_df['Album Name'] == album)
                ].empty
                if has_link:
                    continue
            elif fixer_filter == 'genre':
                prow = predictions_df[(predictions_df['Artist'] == artist) & (predictions_df['Album Name'] == album)]
                if not prow.empty and pd.notna(prow.iloc[0].get('Genres')):
                    continue
            album_options.append(f"{artist} - {album}")

        if fixer_filter:
            st.caption(f"Showing {len(album_options)} albums with issue")

        selected = None
        if not album_options:
            st.success("✅ No albums with this issue!")
        else:
            selected = st.selectbox(
                "Select an album to edit",
                options=album_options,
                key="edit_album_select"
            )

        if selected:
            current_artist = selected.split(' - ')[0]
            current_album = selected.split(' - ', 1)[1]

            # Load predictions fresh
            predictions_df = pd.read_csv(latest_file_path)
            if 'Album' in predictions_df.columns and 'Album Name' not in predictions_df.columns:
                predictions_df['Album Name'] = predictions_df['Album']
            if 'Artist Name(s)' in predictions_df.columns and 'Artist' not in predictions_df.columns:
                predictions_df['Artist'] = predictions_df['Artist Name(s)']

            mask = (
                (predictions_df['Artist'] == current_artist) &
                (predictions_df['Album Name'] == current_album)
            )
            pred_row = predictions_df[mask].iloc[0] if mask.any() else None

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🖼️ Cover Art")
                current_art = album_covers_df[
                    (album_covers_df['Artist'] == current_artist) &
                    (album_covers_df['Album Name'] == current_album)
                ]
                if not current_art.empty and pd.notna(current_art.iloc[0]['Album Art']):
                    st.image(current_art.iloc[0]['Album Art'], width=200)
                else:
                    st.caption("No cover art found")
                new_art = st.text_input("New Artwork URL", key="edit_art_url")
                if st.button("💾 Update Art", key="edit_art_save"):
                    if new_art:
                        local_path = cover_filename(current_artist, current_album)
                        try:
                            import requests as req
                            img_response = req.get(new_art, timeout=10)
                            img_response.raise_for_status()
                            os.makedirs('covers', exist_ok=True)
                            with open(local_path, 'wb') as f:
                                f.write(img_response.content)
                            save_path = local_path
                            st.success("✅ Downloaded!")
                        except Exception as e:
                            save_path = new_art
                            st.warning(f"⚠️ Storing URL instead: {e}")
                        album_covers_df = album_covers_df[~(
                            (album_covers_df['Artist'] == current_artist) &
                            (album_covers_df['Album Name'] == current_album)
                        )]
                        new_row = pd.DataFrame([{'Artist': current_artist, 'Album Name': current_album, 'Album Art': save_path}])
                        album_covers_df = pd.concat([album_covers_df, new_row], ignore_index=True)
                        album_covers_df.to_csv('data/nmf_album_covers.csv', index=False)
                        st.cache_data.clear()
                        st.rerun()

                st.subheader("🔗 Spotify Link")
                current_link = album_links_df[
                    (album_links_df['Artist'] == current_artist) &
                    (album_links_df['Album Name'] == current_album)
                ]
                if not current_link.empty:
                    st.caption(current_link.iloc[0]['Spotify URL'])
                new_link = st.text_input("New Spotify URL", key="edit_spotify_url")
                if st.button("💾 Update Link", key="edit_spotify_save"):
                    if new_link:
                        clean_link = new_link.replace('https://', '').replace('http://', '')
                        album_links_df = album_links_df[~(
                            (album_links_df['Artist'] == current_artist) &
                            (album_links_df['Album Name'] == current_album)
                        )]
                        new_row = pd.DataFrame([{'Artist': current_artist, 'Album Name': current_album, 'Spotify URL': clean_link}])
                        album_links_df = pd.concat([album_links_df, new_row], ignore_index=True)
                        album_links_df.to_csv('data/nmf_album_links.csv', index=False)
                        st.success("Link updated!")
                        st.rerun()

            with col2:
                st.subheader("📝 Metadata")
                if pred_row is not None:
                    new_artist = st.text_input("Artist Name", value=current_artist, key="edit_artist")
                    new_album  = st.text_input("Album Name",  value=current_album,  key="edit_album")
                    new_genres = st.text_input("Genres",      value=str(pred_row.get('Genres', '') or ''), key="edit_genres")
                    new_label  = st.text_input("Label",       value=str(pred_row.get('Label', '') or ''),  key="edit_label")
                    if st.button("💾 Save Metadata", key="edit_metadata_save"):
                        predictions_df.loc[mask, 'Artist']     = new_artist
                        predictions_df.loc[mask, 'Album Name'] = new_album
                        predictions_df.loc[mask, 'Album']      = new_album
                        predictions_df.loc[mask, 'Genres']     = new_genres
                        predictions_df.loc[mask, 'Label']      = new_label
                        predictions_df.to_csv(latest_file_path, index=False)
                        st.success(f"✅ Updated!")
                        st.cache_data.clear()
                        st.rerun()
                else:
                    st.error("Could not find that album in the predictions file.")

    with tab2:
        st.header("☢️ Nuke Albums")
        st.write("Remove albums that shouldn't be in the list (e.g., singles, re-releases).")
        
        nuked_df = load_nuked_albums()
        
        nuke_options = [""] + [f"{r['Artist']} - {r['Album Name']}" for _, r in current_albums.iterrows()]
        nuke_selected = st.selectbox("Select album to nuke", options=nuke_options, key="nuke_select")
        nuke_reason = st.text_input("Reason (e.g., Single, Re-release, Wrong Genre)", key="nuke_reason")
        if nuke_selected and st.button("☢️ Nuke It", key="nuke_btn"):
            nuke_artist = nuke_selected.split(' - ')[0]
            nuke_album = nuke_selected.split(' - ', 1)[1]
            new_nuke = pd.DataFrame([{'Artist': nuke_artist, 'Album Name': nuke_album, 'Reason': nuke_reason}])
            nuked_df = pd.concat([nuked_df, new_nuke], ignore_index=True)
            nuked_df.to_csv('data/nuked_albums.csv', index=False)
            st.success(f"☢️ Nuked {nuke_album}!")
            st.rerun()

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
    album_covers_df = load_album_covers()
    similar_artists_df = load_similar_artists()

    # Load all prediction files
    prediction_files = []
    for file in glob.glob('predictions/*_Album_Recommendations.csv'):
        try:
            date_str = os.path.basename(file).split('_')[0]
            date_obj = datetime.strptime(date_str, '%m-%d-%y')
            prediction_files.append((date_obj, file))
        except:
            continue
    prediction_files.sort(key=lambda x: x[0], reverse=True)
    current_year = datetime.now().year

    # --- HEADER ROW ---
    st.markdown("""
        <style>
        .disc-btn button {
            height: 120px !important;
            font-size: 1.3rem !important;
            font-weight: 600 !important;
            border-radius: 12px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    title_col, btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([2, 1, 1, 1, 1])
    with title_col:
        st.title("📅 Discover")
        st.caption("Pull a random album and rate it.")
    with btn_col1:
        st.markdown('<div class="disc-btn">', unsafe_allow_html=True)
        old_fav_btn = st.button("⭐ Old Fav", use_container_width=True, type="primary", key="old_fav_btn")
        st.caption("Top 50 · gut scored ≥80 · pre-2026")
        st.markdown('</div>', unsafe_allow_html=True)
    with btn_col2:
        st.markdown('<div class="disc-btn">', unsafe_allow_html=True)
        new_potential_btn = st.button("✨ New Potential", use_container_width=True, type="secondary", key="new_potential_btn")
        st.caption("Predicted ≥75 · never rated · 2026 first")
        st.markdown('</div>', unsafe_allow_html=True)
    with btn_col3:
        st.markdown('<div class="disc-btn">', unsafe_allow_html=True)
        take_chance_btn = st.button("🎰 Take A Chance", use_container_width=True, type="secondary", key="take_chance_btn")
        st.caption("Predicted 50–74 · never rated · 2026 first")
        st.markdown('</div>', unsafe_allow_html=True)
    with btn_col4:
        st.markdown('<div class="disc-btn">', unsafe_allow_html=True)
        random_draw_btn = st.button("🎲 Random Draw", use_container_width=True, type="secondary", key="random_draw_btn")
        st.caption("Anything goes · any week · any score")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- OLD FAV LOGIC ---
    if old_fav_btn:
        candidates = []

        try:
            training_df = pd.read_csv('data/2026_training_complete_with_features.csv')

            # Top 50 = top_100_ranked with liked >= 80
            top50 = training_df[
                (training_df['source_type'] == 'top_100_ranked') &
                (training_df['liked'] >= 80)
            ][['Album Name', 'Artist Name(s)']].drop_duplicates()
            for _, row in top50.iterrows():
                candidates.append({
                    'Album': row['Album Name'],
                    'Artist': str(row['Artist Name(s)']).split(';')[0].split(',')[0].strip(),
                    'source': 'Top 50'
                })
        except Exception as e:
            st.warning(f"Could not load training data: {e}")

        try:
            master_file = 'feedback/master_gut_scores.csv'
            if os.path.exists(master_file):
                gut_df = pd.read_csv(master_file)
                high_gut = gut_df[
                    (gut_df['gut_score'] >= 80) &
                    (pd.to_datetime(gut_df['gut_score_date'], errors='coerce').dt.year < current_year)
                ]
                for _, row in high_gut.iterrows():
                    candidates.append({
                        'Album': row['Album'],
                        'Artist': row['Artist'],
                        'source': f"Gut Score ({int(row['gut_score'])})"
                    })
        except Exception as e:
            st.warning(f"Could not load gut scores: {e}")

        if not candidates:
            st.error("No Old Favs found.")
        else:
            pick = random.choice(candidates)
            st.session_state['disc_pick'] = pick
            st.session_state['disc_mode'] = 'old_fav'

    # --- NEW POTENTIAL LOGIC ---
    elif new_potential_btn:
        hidden_gems_df = load_hidden_gems_cache()

        # Check for new prediction files not yet in cache and append them
        try:
            cache_file = 'data/hidden_gems_cache.csv'
            cache_mtime = os.path.getmtime(cache_file) if os.path.exists(cache_file) else 0
            training_df, master_df = load_training_history()
            new_gems = []

            for f in glob.glob('predictions/*_Album_Recommendations.csv'):
                if os.path.getmtime(f) <= cache_mtime:
                    continue
                # This prediction file is newer than the cache — scan it
                try:
                    df = pd.read_csv(f)
                    if 'Album' in df.columns and 'Album Name' not in df.columns:
                        df['Album Name'] = df['Album']
                    elif 'Album Name' in df.columns and 'Album' not in df.columns:
                        df['Album'] = df['Album Name']
                    if 'Artist Name(s)' in df.columns and 'Artist' not in df.columns:
                        df['Artist'] = df['Artist Name(s)']
                    score_col = 'Predicted_Score' if 'Predicted_Score' in df.columns else 'avg_score'
                    date_str = os.path.basename(f).split('_')[0]
                    date_obj = datetime.strptime(date_str, '%m-%d-%y')

                    for _, row in df.drop_duplicates(subset=['Artist', 'Album']).iterrows():
                        score = row.get(score_col, 0)
                        if pd.isna(score) or score < 75:
                            continue
                        artist = str(row.get('Artist', '')).strip()
                        album = str(row.get('Album', '')).strip()
                        if not artist or not album:
                            continue
                        # Skip if already in training data or gut scored
                        in_training = not training_df[
                            (training_df['Album Name'] == album) &
                            (training_df['Artist Name(s)'].str.contains(artist, na=False))
                        ].empty
                        in_gut = not master_df.empty and not master_df[
                            (master_df['Album'] == album) &
                            (master_df['Artist'] == artist)
                        ].empty
                        if not in_training and not in_gut:
                            new_gems.append({
                                'Artist': artist,
                                'Album': album,
                                'Predicted_Score': float(score),
                                'Source_Week': date_obj.strftime('%Y-%m-%d'),
                                'Genres': row.get('Genres', ''),
                                'Label': row.get('Label', '')
                            })
                except:
                    continue

            if new_gems:
                new_df = pd.DataFrame(new_gems).drop_duplicates(subset=['Artist', 'Album'])
                hidden_gems_df = pd.concat([hidden_gems_df, new_df], ignore_index=True)
                hidden_gems_df = hidden_gems_df.drop_duplicates(subset=['Artist', 'Album'], keep='last')
                hidden_gems_df.to_csv(cache_file, index=False)
                load_hidden_gems_cache.clear()
                st.toast(f"✨ Found {len(new_gems)} new Hidden Gems from recent weeks!")
        except Exception as e:
            pass

        if hidden_gems_df.empty:
            st.error("No Hidden Gems cache found. Run update in Settings first.")
        else:
            # Bias toward current year first
            if 'Source_Week' in hidden_gems_df.columns:
                this_year = hidden_gems_df[
                    pd.to_datetime(hidden_gems_df['Source_Week'], errors='coerce').dt.year == current_year
                ]
                older = hidden_gems_df[
                    pd.to_datetime(hidden_gems_df['Source_Week'], errors='coerce').dt.year < current_year
                ]
                pool = this_year if not this_year.empty else older
            else:
                pool = hidden_gems_df

            # Filter out already gut-scored albums
            try:
                master_file = 'feedback/master_gut_scores.csv'
                if os.path.exists(master_file):
                    gut_df = pd.read_csv(master_file)
                    gut_scored = set(zip(gut_df['Artist'], gut_df['Album']))
                    pool = pool[~pool.apply(
                        lambda r: (r.get('Artist', ''), r.get('Album', r.get('Album Name', ''))) in gut_scored,
                        axis=1
                    )]
            except Exception as e:
                pass

            if pool.empty:
                st.error("All Hidden Gems have been gut scored — you're caught up! 🎉")
            else:
                pick_row = pool.sample(n=1).iloc[0]
                st.session_state['disc_pick'] = {
                    'Album': pick_row.get('Album', pick_row.get('Album Name', '')),
                    'Artist': pick_row.get('Artist', ''),
                    'Predicted_Score': pick_row.get('Predicted_Score', None) or None,
                    'Genres': pick_row.get('Genres', ''),
                    'source': 'New Potential'
                }
                st.session_state['disc_mode'] = 'new_potential'

    # --- TAKE A CHANCE LOGIC ---
    elif take_chance_btn:
        try:
            training_df, master_df = load_training_history()
            gut_scored = set(zip(master_df['Artist'], master_df['Album'])) if not master_df.empty else set()
            top_rated = set(training_df[
                training_df['source_type'].isin(['top_100_ranked', 'honorable_mention'])
            ]['Album Name'].tolist())
        except:
            gut_scored = set()
            top_rated = set()

        candidates = []
        for date_obj, file in prediction_files:
            try:
                df = pd.read_csv(file)
                album_col = 'Album' if 'Album' in df.columns else 'Album Name'
                artist_col = 'Artist' if 'Artist' in df.columns else 'Artist Name(s)'
                score_col = 'Predicted_Score' if 'Predicted_Score' in df.columns else 'avg_score'
                for _, row in df.drop_duplicates(subset=[artist_col, album_col]).iterrows():
                    score = row.get(score_col, 0)
                    if pd.isna(score) or not (50 <= float(score) <= 74):
                        continue
                    artist = str(row.get(artist_col, '')).strip()
                    album = str(row.get(album_col, '')).strip()
                    if not artist or not album:
                        continue
                    if (artist, album) in gut_scored:
                        continue
                    if album in top_rated:
                        continue
                    candidates.append({
                        'Artist': artist,
                        'Album': album,
                        'Predicted_Score': float(score),
                        'Genres': row.get('Genres', ''),
                        'Label': row.get('Label', ''),
                        'source': date_obj.strftime('%B %d, %Y'),
                        'date_obj': date_obj
                    })
            except:
                continue

        if not candidates:
            st.error("No candidates found.")
        else:
            # Bias toward current year
            this_year = [c for c in candidates if c['date_obj'].year == current_year]
            pool = this_year if this_year else candidates
            pick = random.choice(pool)
            pick.pop('date_obj', None)
            st.session_state['disc_pick'] = pick
            st.session_state['disc_mode'] = 'take_chance'

    # --- RANDOM DRAW LOGIC ---
    elif random_draw_btn:
        all_albums = []
        for date_obj, file in prediction_files:
            try:
                df = pd.read_csv(file)
                artist_col = 'Artist' if 'Artist' in df.columns else 'Artist Name(s)'
                album_col = 'Album' if 'Album' in df.columns else 'Album Name'
                for _, row in df[[artist_col, album_col]].drop_duplicates().iterrows():
                    all_albums.append({
                        'Artist': str(row[artist_col]).strip(),
                        'Album': str(row[album_col]).strip(),
                        'Predicted_Score': row.get('Predicted_Score', row.get('avg_score', 0)),
                        'Genres': row.get('Genres', ''),
                        'Label': row.get('Label', ''),
                        'source': date_obj.strftime('%B %d, %Y')
                    })
            except:
                continue

        if not all_albums:
            st.error("No albums found.")
        else:
            pick = random.choice(all_albums)
            st.session_state['disc_pick'] = pick
            st.session_state['disc_mode'] = 'random_draw'

    # --- RENDER PICKED ALBUM ---
    if st.session_state.get('disc_pick'):
        pick = st.session_state['disc_pick']
        artist = pick['Artist']
        album = pick['Album']
        mode = st.session_state.get('disc_mode', '')

        st.caption(f"{'⭐ Old Fav' if mode == 'old_fav' else '✨ New Potential'} · {pick.get('source', '')}")

        # For Old Fav — look up the actual liked score from training data
        gut_prefill = None
        if mode == 'old_fav':
            try:
                training_df, _ = load_training_history()
                match = training_df[
                    (training_df['Album Name'] == album) &
                    (training_df['Artist Name(s)'].str.contains(artist, na=False))
                ]
                if not match.empty:
                    gut_prefill = float(match.iloc[0]['liked'])
            except:
                pass

        # For Old Fav — pull Label and Genres from training data too
        training_label = ''
        training_genres = ''
        if mode == 'old_fav':
            try:
                training_df, _ = load_training_history()
                match = training_df[
                    (training_df['Album Name'] == album) &
                    (training_df['Artist Name(s)'].str.contains(artist, na=False))
                ]
                if not match.empty:
                    training_label = str(match.iloc[0].get('Record Label', '')) if pd.notna(match.iloc[0].get('Record Label')) else ''
                    training_genres = str(match.iloc[0].get('Genres', '')) if pd.notna(match.iloc[0].get('Genres')) else ''
            except:
                pass

        # Build a single-row dataframe that display_album_predictions can render
        row_data = {
            'Artist': artist,
            'Album': album,
            'Album Name': album,
            'avg_score': gut_prefill or 0,
            'Predicted_Score': gut_prefill or 0,
            'Genres': training_genres or pick.get('Genres', ''),
            'Label': training_label or pick.get('Label', ''),
            'gut_score': gut_prefill,
        }
        single_df = pd.DataFrame([row_data])

        # Find the source prediction file for gut score saving
        # Also look up real predicted score if missing from cache
        source_file = None
        real_score = pick.get('Predicted_Score') or 0
        for date_obj, file in prediction_files:
            try:
                df = pd.read_csv(file)
                album_col = 'Album' if 'Album' in df.columns else 'Album Name'
                artist_col = 'Artist' if 'Artist' in df.columns else 'Artist Name(s)'
                match = df[(df[album_col] == album) & (df[artist_col].str.contains(artist, na=False))]
                if not match.empty:
                    source_file = file
                    if real_score == 0:
                        score_col = 'Predicted_Score' if 'Predicted_Score' in match.columns else 'avg_score'
                        real_score = float(match.iloc[0].get(score_col, 0) or 0)
                    break
            except:
                continue

        row_data['avg_score'] = real_score
        row_data['Predicted_Score'] = real_score
        single_df = pd.DataFrame([row_data])

        display_album_predictions(single_df, album_covers_df, similar_artists_df, source_file)

def dacus_game_page(G):
    st.title("🎵 6 Degrees of Lucy Dacus")

    tab_dacus, tab_any = st.tabs(["🎵 6 Degrees of Lucy Dacus", "🔗 Any Two Artists"])

    all_artists = sorted(list(G.nodes()))

    def artist_search_and_select(label_prefix, key_suffix):
        search_term = st.text_input(f"Search for {label_prefix}:", key=f"search_{key_suffix}")
        if search_term:
            filtered = [a for a in all_artists if search_term.lower() in a.lower()]
            if not filtered:
                st.warning(f"No artists found matching '{search_term}'")
                return None
            if len(filtered) > 10:
                st.info(f"Found {len(filtered)} matches. Showing top 10.")
                filtered = filtered[:10]
            return st.selectbox("Select an artist:", options=filtered, key=f"select_{key_suffix}")
        else:
            popular_artists = [a for a in ["Phoebe Bridgers", "Boygenius", "Julien Baker",
                               "Japanese Breakfast", "Mitski", "Big Thief",
                               "The National", "Snail Mail", "Soccer Mommy"] if a in G.nodes()]
            selected = st.selectbox(
                "Or pick a popular artist:",
                options=popular_artists + ["Select an artist..."],
                index=len(popular_artists),
                key=f"popular_{key_suffix}"
            )
            return None if selected == "Select an artist..." else selected

    def show_path(path, G):
        st.subheader("Network Path Visualization")
        with st.spinner("Generating network visualization..."):
            path_nodes = set(path)
            context_nodes = set()
            for node in path:
                context_nodes.update(list(G.neighbors(node))[:3])
            subgraph = G.subgraph(path_nodes.union(context_nodes))
            fig = visualize_artist_network(subgraph, path)
            st.plotly_chart(fig, use_container_width=True)

    with tab_dacus:
        st.write("""
        ### How It Works
        Select an artist to see how closely they're connected to Lucy Dacus!
        The **Dacus number** is the number of connections between the artist and Lucy Dacus.
        """)
        selected_artist = artist_search_and_select("an artist", "dacus")
        if selected_artist:
            dacus_number, path = calculate_dacus_number(selected_artist, G)
            if dacus_number is not None:
                st.success(f"**Dacus Number:** {dacus_number}")
                st.write(f"**Path to Lucy Dacus:** {' → '.join(path)}")
                show_path(path, G)
            else:
                st.error("No path found. This artist might not be connected to Lucy Dacus in our network.")
        else:
            st.info("Please search for an artist or select one from the list")

    with tab_any:
        st.write("""
        ### How It Works
        Pick any two artists and see how they're connected in your music universe!
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Artist A**")
            artist_a = artist_search_and_select("Artist A", "any_a")
        with col2:
            st.markdown("**Artist B**")
            artist_b = artist_search_and_select("Artist B", "any_b")

        if artist_a and artist_b:
            if artist_a == artist_b:
                st.warning("Pick two different artists!")
            else:
                try:
                    path = nx.shortest_path(G, source=artist_a, target=artist_b)
                    distance = len(path) - 1
                    st.success(f"**Distance:** {distance} degrees")
                    st.write(f"**Path:** {' → '.join(path)}")
                    show_path(path, G)
                except nx.NetworkXNoPath:
                    st.error(f"No path found between {artist_a} and {artist_b} in your music universe.")
        else:
            st.info("Search for two artists above to find their connection.")
        
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
    # Hide sidebar and tighten top padding
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Lora:ital@1&display=swap');
        [data-testid="stSidebar"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
        .block-container {padding-top: 1rem !important;}
        </style>
    """, unsafe_allow_html=True)

    # Load data needed for multiple pages
    album_covers_df = load_album_covers()
    similar_artists_df = load_similar_artists()
    liked_similar_df = load_similar_artists()

    # Tab navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎵 Weekly Predictions",
        "📅 Historical Ratings",
        "📓 The Model",
        "🎵 6 Degrees of Lucy Dacus",
        "👤 About Me",
        "🛠️ Album Fixer",
    ])

    with tab1:
        prediction_files = get_all_prediction_files()
        if not prediction_files:
            st.error("No prediction files found!")
            return

        options = {formatted_date: file_path for file_path, _, formatted_date in prediction_files}
        most_recent = prediction_files[0][2]
        recent_file = prediction_files[0][0]

        if 'selected_week' in st.session_state and st.session_state['selected_week'] in options:
            selected_date = st.session_state['selected_week']
            selected_file = options[selected_date]
        else:
            selected_date = most_recent
            selected_file = recent_file

        # Title row — left: title, right: week selector
        title_col, week_col = st.columns([3, 1])
        with title_col:
            st.title("🎵 New Music Friday Regression Model")
            st.subheader("Personalized New Music Friday Recommendations")
        with week_col:
            st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
            week_options = [most_recent] + [date for date in options.keys() if date != most_recent]
            top_selected = st.selectbox(
                "📅 Week",
                options=week_options,
                index=week_options.index(selected_date) if selected_date in week_options else 0,
                format_func=lambda x: "📅 " + x,
                key="top_week_selector"
            )
            if top_selected != selected_date:
                st.session_state['selected_week'] = top_selected
                st.rerun()

        predictions_df, analysis_date = load_predictions(selected_file)
        if predictions_df is None:
            st.error("No data found for selected week.")
            return
        analysis_date_formatted = datetime.strptime(analysis_date, '%Y-%m-%d').strftime('%B %d, %Y')
        st.subheader(f"Showing {len(predictions_df)} albums for {analysis_date_formatted}")
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

        col2, col3 = st.columns(2)
        with col2:
            if 'Genres' in pd.read_csv(recent_file).columns:
                temp_df = pd.read_csv(recent_file)
                all_genres = set()
                for genres in temp_df['Genres'].dropna():
                    for g in str(genres).split(','):
                        all_genres.add(g.strip())
                sorted_genres = sorted(list(all_genres))
                selected_genres = st.multiselect("🎵 Filter by Genre", options=sorted_genres)
            else:
                selected_genres = []
        with col3:
            # Build list of unique artist–album pairs for the current week
            album_pairs = predictions_df[['Artist', 'Album Name']].drop_duplicates()
            album_options = ["All albums"] + [f"{row['Artist']} - {row['Album Name']}" for _, row in album_pairs.iterrows()]
            selected_album = st.selectbox(
                "🔍 Select album",
                album_options,
                index=0,
                key=f"album_select_{analysis_date_formatted}"  # resets when week changes
            )
        
        st.markdown("<div style='margin-top: -30px;'></div>", unsafe_allow_html=True)
        sort_order = "Highest First"
        if selected_genres:
            predictions_df = predictions_df[
                predictions_df['Genres'].apply(lambda x: any(g.strip() in selected_genres for g in str(x).split(',')) if pd.notna(x) else False)
            ]
        if selected_album != "All albums":
            artist, album = selected_album.split(" - ", 1)
            predictions_df = predictions_df[(predictions_df['Artist'] == artist) & (predictions_df['Album Name'] == album)]
        score_col = 'Predicted_Score' if 'Predicted_Score' in predictions_df.columns else 'avg_score'
        predictions_df = predictions_df.sort_values(by=score_col, ascending=False)
        display_album_predictions(predictions_df, album_covers_df, similar_artists_df, selected_file)

        # Silently pre-warm the next oldest week in cache
        current_index = next((i for i, (f, d, date) in enumerate(prediction_files) if date == selected_date), None)
        if current_index is not None and current_index + 1 < len(prediction_files):
            next_file = prediction_files[current_index + 1][0]
            load_predictions(next_file)

        st.caption(f"📚 {len(prediction_files)} weeks available — use the week selector above to browse")

    with tab2:
        historical_rating_page()

    with tab3:
        notebook_page()

    with tab4:
        latest_file = get_all_prediction_files()[0][0] if get_all_prediction_files() else None
        predictions_data = load_predictions(latest_file)
        if predictions_data is None:
            st.error("Could not load prediction data.")
            return
        df, _ = predictions_data
        G = build_graph(df, liked_similar_df, include_nmf=True)
        dacus_game_page(G)

    with tab5:
        about_me_page()

    with tab6:
        album_fixer_page()

if __name__ == "__main__":
    main()