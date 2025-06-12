import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
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
import sys
import traceback

# HuggingFace Spaces Compatibility Functions
def ensure_directories_exist():
    """
    Ensure all required directories exist, creating them if necessary.
    This is important for HuggingFace Spaces where directories might not be pre-created.
    """
    required_dirs = [
        'data',
        'predictions', 
        'feedback',
        'graphics'
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")

def safe_load_csv(file_path, default_columns=None):
    """
    Safely load a CSV file with error handling.
    Returns an empty DataFrame with default columns if the file doesn't exist or can't be read.
    """
    try:
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            st.warning(f"File not found: {file_path}")
            if default_columns:
                return pd.DataFrame(columns=default_columns)
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        if default_columns:
            return pd.DataFrame(columns=default_columns)
        return pd.DataFrame()

# Initialize directories and error handling
try:
    # Ensure all required directories exist
    ensure_directories_exist() 

# Check if App is Running Locally or on Streamlit's Servers
def is_running_on_streamlit():
    return os.getenv("STREAMLIT_SERVER_RUNNING", "False").lower() == "true"

# Use this flag to control feedback buttons
IS_LOCAL = not is_running_on_streamlit()

@st.cache_data
def load_nuked_albums():
    """
    Load the list of nuked albums from the CSV file.
    """
    nuked_albums_file = 'data/nuked_albums.csv'
    return safe_load_csv(nuked_albums_file, ['Artist', 'Album Name', 'Reason'])

st.set_page_config(
    page_title="New Music Friday Regression Model",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    # Ensure 'playlist_origin' column exists (silently add if missing)
    if 'playlist_origin' not in predictions_df.columns:
        predictions_df['playlist_origin'] = 'unknown'  # Default value
    
    # Ensure 'Artist Name(s)' column exists (silently add if missing)
    if 'Artist Name(s)' not in predictions_df.columns:
        predictions_df['Artist Name(s)'] = 'Unknown Artist'  # Default value
    
    # Remove duplicate albums if any
    predictions_df = predictions_df.drop_duplicates(subset=['Artist', 'Album Name'], keep='first')
    
    date_str = os.path.basename(file_path).split('_')[0]
    analysis_date = datetime.strptime(date_str, '%m-%d-%y').strftime('%Y-%m-%d')
    
    return predictions_df, analysis_date

@st.cache_data
def load_album_covers():
    return safe_load_csv('data/nmf_album_covers.csv', ['Artist', 'Album Name', 'Album Art'])

@st.cache_data
def load_album_links():
    return safe_load_csv('data/nmf_album_links.csv', ['Album Name', 'Artist Name(s)', 'Spotify URL'])

@st.cache_data
def load_similar_artists():
    return safe_load_csv('data/nmf_similar_artists.csv', ['Artist', 'Similar Artists'])

@st.cache_data
def load_liked_similar():
    """
    Load the dataset of similar artists for liked artists.
    """
    return safe_load_csv('data/liked_artists_only_similar.csv', ['Artist', 'Similar Artists'])

def load_training_data():
    df = safe_load_csv('data/df_cleaned_pre_standardized.csv')
    if not df.empty and 'playlist_origin' in df.columns:
        return df[df['playlist_origin'] != 'df_nmf'].copy()
    return df

# Improved feedback functions with better error handling
def save_feedback(album_name, artist, feedback, review=None):
    """
    Save user feedback on albums with improved error handling for HuggingFace Spaces.
    """
    feedback_dir = 'feedback'
    feedback_file = os.path.join(feedback_dir, 'feedback.csv')
    
    # Ensure the feedback directory exists
    if not os.path.exists(feedback_dir):
        try:
            os.makedirs(feedback_dir, exist_ok=True)
        except Exception as e:
            st.error(f"Error creating feedback directory: {e}")
            return
    
    # Create a dataframe with the new feedback
    new_feedback = pd.DataFrame({
        'Album Name': [album_name],
        'Artist': [artist],
        'Feedback': [feedback],
        'Review': [review if review else ""]
    })

    # Load existing feedback if file exists
    if os.path.exists(feedback_file):
        try:
            existing_feedback = pd.read_csv(feedback_file, quoting=1)
            
            # Remove existing feedback for this album and artist
            existing_feedback = existing_feedback[
                ~((existing_feedback['Album Name'] == album_name) & 
                  (existing_feedback['Artist'] == artist))
            ]
            
            # Combine with new feedback
            combined_feedback = pd.concat([existing_feedback, new_feedback], ignore_index=True)
        except Exception as e:
            st.warning(f"Error reading existing feedback: {e}. Creating new file.")
            combined_feedback = new_feedback
    else:
        combined_feedback = new_feedback
    
    try:
        # Save with proper quoting to handle commas in fields
        combined_feedback.to_csv(feedback_file, index=False, quoting=1)
        
        # Clear cache after saving feedback
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

def load_feedback():
    feedback_file = 'feedback/feedback.csv'
    return safe_load_csv(feedback_file, ['Album Name', 'Artist', 'Feedback', 'Review'])

def load_public_feedback():
    feedback_file = 'feedback/public_feedback.csv'
    return safe_load_csv(feedback_file, ['Album Name', 'Artist', 'Feedback', 'Username', 'Timestamp', 'Review'])

# Updated feedback functions
def save_feedback(album_name, artist, feedback, review=None):
    feedback_file = 'feedback/feedback.csv'
    if not os.path.exists('feedback'):
        os.makedirs('feedback')
    
    # Create a dataframe with the new feedback
    new_feedback = pd.DataFrame({
        'Album Name': [album_name],
        'Artist': [artist],
        'Feedback': [feedback],
        'Review': [review if review else ""]
    })

    # Load existing feedback if file exists
    if os.path.exists(feedback_file):
        try:
            # Use proper quoting and escape characters when reading
            existing_feedback = pd.read_csv(feedback_file, quoting=1)  # QUOTE_ALL
            
            # Remove existing feedback for this album and artist
            existing_feedback = existing_feedback[
                ~((existing_feedback['Album Name'] == album_name) & 
                  (existing_feedback['Artist'] == artist))
            ]
            
            # Combine with new feedback
            combined_feedback = pd.concat([existing_feedback, new_feedback], ignore_index=True)
        except Exception as e:
            st.warning(f"Error reading existing feedback: {e}. Creating new file.")
            # If reading fails, start fresh with just the new feedback
            combined_feedback = new_feedback
    else:
        combined_feedback = new_feedback
    
    # Save with proper quoting to handle commas in fields
    combined_feedback.to_csv(feedback_file, index=False, quoting=1)  # QUOTE_ALL
    
    # Clear cache after saving feedback
    st.cache_data.clear()

def save_public_feedback(album_name, artist, feedback, username="Anonymous", review=None):
    feedback_dir = 'feedback'
    feedback_file = os.path.join(feedback_dir, 'public_feedback.csv')
    
    # Ensure the feedback directory exists
    if not os.path.exists(feedback_dir):
        try:
            os.makedirs(feedback_dir, exist_ok=True)
        except Exception as e:
            st.error(f"Error creating feedback directory: {e}")
            return
    
    # Create a dataframe with the new feedback
    new_feedback = pd.DataFrame({
        'Album Name': [album_name],
        'Artist': [artist],
        'Feedback': [feedback],
        'Username': [username],
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'Review': [review if review else ""]
    })

    # Load existing feedback if file exists
    if os.path.exists(feedback_file):
        try:
            existing_feedback = pd.read_csv(feedback_file, quoting=1)
            combined_feedback = pd.concat([existing_feedback, new_feedback], ignore_index=True)
        except Exception as e:
            st.warning(f"Error reading existing public feedback: {e}. Creating new file.")
            combined_feedback = new_feedback
    else:
        combined_feedback = new_feedback
    
    try:
        # Save with proper quoting to handle commas in fields
        combined_feedback.to_csv(feedback_file, index=False, quoting=1)
        
        # Clear cache after saving feedback
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Error saving public feedback: {e}")

# Updated load feedback functions
def load_feedback():
    feedback_file = 'feedback/feedback.csv'
    if os.path.exists(feedback_file):
        try:
            # Use quoting=1 (QUOTE_ALL) to properly handle commas in fields
            return pd.read_csv(feedback_file, quoting=1)
        except Exception as e:
            st.warning(f"Error loading feedback data: {e}")
            # Try to recover the file
            try:
                # Attempt to read with different options
                df = pd.read_csv(feedback_file, quoting=1, error_bad_lines=False) 
                st.info("Partially recovered feedback data")
                return df
            except:
                # If all recovery attempts fail, provide an empty DataFrame as fallback
                st.error("Could not recover feedback data. Starting with fresh feedback file.")
                # Backup the problematic file
                if os.path.exists(feedback_file):
                    backup_file = feedback_file + ".backup." + datetime.now().strftime("%Y%m%d%H%M%S")
                    try:
                        os.rename(feedback_file, backup_file)
                        st.info(f"Backed up problematic feedback file to {backup_file}")
                    except:
                        pass
                return pd.DataFrame(columns=['Album Name', 'Artist', 'Feedback', 'Review'])
    return pd.DataFrame(columns=['Album Name', 'Artist', 'Feedback', 'Review'])

def load_public_feedback():
    feedback_file = 'feedback/public_feedback.csv'
    if os.path.exists(feedback_file):
        try:
            # Use quoting=1 (QUOTE_ALL) to properly handle commas in fields
            return pd.read_csv(feedback_file, quoting=1)
        except Exception as e:
            st.warning(f"Error loading public feedback data: {e}")
            # Try to recover the file
            try:
                # Attempt to read with different options
                df = pd.read_csv(feedback_file, quoting=1, error_bad_lines=False) 
                st.info("Partially recovered public feedback data")
                return df
            except:
                # If all recovery attempts fail, provide an empty DataFrame as fallback
                st.error("Could not recover public feedback data. Starting with fresh feedback file.")
                # Backup the problematic file
                if os.path.exists(feedback_file):
                    backup_file = feedback_file + ".backup." + datetime.now().strftime("%Y%m%d%H%M%S")
                    try:
                        os.rename(feedback_file, backup_file)
                        st.info(f"Backed up problematic feedback file to {backup_file}")
                    except:
                        pass
                return pd.DataFrame(columns=['Album Name', 'Artist', 'Feedback', 'Username', 'Timestamp', 'Review'])
    return pd.DataFrame(columns=['Album Name', 'Artist', 'Feedback', 'Username', 'Timestamp', 'Review'])

def get_public_feedback_stats(album_name, artist):
    """Get statistics for public feedback on a specific album"""
    public_feedback_df = load_public_feedback()
    
    # Filter for this album
    album_feedback = public_feedback_df[
        (public_feedback_df['Album Name'] == album_name) & 
        (public_feedback_df['Artist'] == artist)
    ]
    
    if album_feedback.empty:
        return {"like": 0, "mid": 0, "dislike": 0, "total": 0}
    
    # Count each feedback type
    feedback_counts = album_feedback['Feedback'].value_counts().to_dict()
    
    # Ensure all categories exist
    stats = {
        "like": feedback_counts.get('like', 0),
        "mid": feedback_counts.get('mid', 0),
        "dislike": feedback_counts.get('dislike', 0),
        "total": len(album_feedback)
    }
    
    return stats

def get_recent_public_feedback(album_name, artist, limit=3):
    """Get the most recent public feedback for a specific album"""
    public_feedback_df = load_public_feedback()
    
    # Filter for this album
    album_feedback = public_feedback_df[
        (public_feedback_df['Album Name'] == album_name) & 
        (public_feedback_df['Artist'] == artist)
    ]
    
    if album_feedback.empty:
        return pd.DataFrame(columns=['Album Name', 'Artist', 'Feedback', 'Username', 'Timestamp', 'Review'])
    
    # Sort by timestamp (newest first) and take the top 'limit' entries
    album_feedback['Timestamp'] = pd.to_datetime(album_feedback['Timestamp'])
    recent_feedback = album_feedback.sort_values('Timestamp', ascending=False).head(limit)
    
    return recent_feedback

# The display_album_predictions function
def display_album_predictions(filtered_data, album_covers_df, similar_artists_df):
    try:
        album_links_df = load_album_links()
    except Exception as e:
        st.error(f"Error loading album links: {e}")
        album_links_df = pd.DataFrame()
    
    try:
        merged_data = filtered_data.merge(
            album_covers_df[['Artist', 'Album Name', 'Album Art']], 
            on=['Artist', 'Album Name'],
            how='left'
        )
        
        if not album_links_df.empty:
            merged_data = merged_data.merge(
                album_links_df[['Album Name', 'Artist Name(s)', 'Spotify URL']],
                left_on=['Album Name', 'Artist'],
                right_on=['Album Name', 'Artist Name(s)'],
                how='left'
            )
    except Exception as e:
        st.error(f"Error merging data: {e}")
        merged_data = filtered_data
    
    filtered_albums = merged_data
    
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
                st.markdown(f'<div class="album-title" style="font-size: 1.8rem; font-weight: 600; margin-bottom: 16px;">{row["Artist"]} - {row["Album Name"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;"><strong>Genre:</strong> {row["Genres"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;"><strong>Label:</strong> {row["Label"]}</div>', unsafe_allow_html=True)
                
                similar_artists = similar_artists_df[
                    similar_artists_df['Artist'] == row['Artist']
                ]
                
                if not similar_artists.empty:
                    similar_list = similar_artists.iloc[0]['Similar Artists']
                    st.markdown(f'<div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;"><strong>Similar Artists:</strong> {similar_list}</div>', unsafe_allow_html=True)
                
                if 'Spotify URL' in row and pd.notna(row['Spotify URL']):
                    spotify_url = row['Spotify URL']
                    st.markdown(f'''
                        <a href="https://{spotify_url}" target="_blank" class="spotify-button">
                            ▶ Play on Spotify
                        </a>
                    ''', unsafe_allow_html=True) 
                
                # Public rating section with username input
                st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                st.markdown('<div style="font-weight: 600; margin-bottom: 8px;">Mike wants to know what you think!</div>', unsafe_allow_html=True)

                # Username input
                username = st.text_input("Your name (optional):", key=f"username_input_{idx}", value="")
                username = username.strip() if username else "Anonymous"

                # Load existing feedback to pre-populate the review field
                feedback_df = load_feedback()
                existing_feedback = feedback_df[
                    (feedback_df['Album Name'] == row['Album Name']) & 
                    (feedback_df['Artist'] == row['Artist'])
                ]
                
                # Create a unique key using album name and artist
                unique_key = f"{row['Album Name']}_{row['Artist']}"

                # Pre-populate the review field if feedback exists
                # But set value to empty string to prevent persistence
                existing_review = ""
                if not existing_feedback.empty:
                    existing_review = existing_feedback.iloc[0].get('Review', '')
                    # We still load it to display elsewhere, but don't use it in the text area

                # Add review input field with empty value to prevent persistence
                review = st.text_area("Mini review (optional):", 
                                     value="", 
                                     key=f"review_input_{unique_key}", 
                                     max_chars=200, 
                                     height=80)


                # Create fixed-width columns for buttons
                button_cols = st.columns(3)

                # Create a unique key using album name and artist
                unique_key = f"{row['Album Name']}_{row['Artist']}"

                # Like Button
                with button_cols[0]:
                    if st.button('👍 Like', key=f"public_like_{unique_key}", use_container_width=True):
                        if username == "Mike S":
                            # Load existing feedback to check if a review already exists
                            feedback_df = load_feedback()
                            existing_feedback = feedback_df[
                                (feedback_df['Album Name'] == row['Album Name']) & 
                                (feedback_df['Artist'] == row['Artist'])
                            ]
                            
                            # Preserve existing review if it exists
                            existing_review = ""
                            if not existing_feedback.empty:
                                existing_review = existing_feedback.iloc[0].get('Review', '')
                            
                            # Save feedback with the new review (if provided) or the existing review
                            save_feedback(row['Album Name'], row['Artist'], 'like', review or existing_review)
                            # Display as just "Mike"
                            username = "Mike"
                        else:
                            save_public_feedback(row['Album Name'], row['Artist'], 'like', username, review)
                        
                        # Just rerun - the review will be cleared because we set value=""
                        st.rerun()

                # Mid Button
                with button_cols[1]:
                    if st.button('😐 Mid', key=f"public_mid_{unique_key}", use_container_width=True):
                        if username == "Mike S":
                            # Load existing feedback to check if a review already exists
                            feedback_df = load_feedback()
                            existing_feedback = feedback_df[
                                (feedback_df['Album Name'] == row['Album Name']) & 
                                (feedback_df['Artist'] == row['Artist'])
                            ]
                            
                            # Preserve existing review if it exists
                            existing_review = ""
                            if not existing_feedback.empty:
                                existing_review = existing_feedback.iloc[0].get('Review', '')
                            
                            # Save feedback with the new review (if provided) or the existing review
                            save_feedback(row['Album Name'], row['Artist'], 'mid', review or existing_review)
                            # Display as just "Mike"
                            username = "Mike"
                        else:
                            save_public_feedback(row['Album Name'], row['Artist'], 'mid', username, review)
                        
                        # Just rerun - the review will be cleared because we set value=""
                        st.rerun()

                # Dislike Button
                with button_cols[2]:
                    if st.button('👎 Dislike', key=f"public_dislike_{unique_key}", use_container_width=True):
                        if username == "Mike S":
                            # Load existing feedback to check if a review already exists
                            feedback_df = load_feedback()
                            existing_feedback = feedback_df[
                                (feedback_df['Album Name'] == row['Album Name']) & 
                                (feedback_df['Artist'] == row['Artist'])
                            ]
                            
                            # Preserve existing review if it exists
                            existing_review = ""
                            if not existing_feedback.empty:
                                existing_review = existing_feedback.iloc[0].get('Review', '')
                            
                            # Save feedback with the new review (if provided) or the existing review
                            save_feedback(row['Album Name'], row['Artist'], 'dislike', review or existing_review)
                            # Display as just "Mike"
                            username = "Mike"
                        else:
                            save_public_feedback(row['Album Name'], row['Artist'], 'dislike', username, review)
                        
                        # Just rerun - the review will be cleared because we set value=""
                        st.rerun()



                st.markdown('</div>', unsafe_allow_html=True)  # Close the feedback-container div
                
                # Display public rating stats
                public_stats = get_public_feedback_stats(row['Album Name'], row['Artist'])
                if public_stats['total'] > 0:
                    recent_feedback = get_recent_public_feedback(row['Album Name'], row['Artist'], 3)
                    feedback_display = ""
                    for _, fb in recent_feedback.iterrows():
                        emoji = "👍" if fb['Feedback'] == 'like' else "😐" if fb['Feedback'] == 'mid' else "👎"
                        feedback_display += f"{fb['Username']} {emoji} • "
                    
                    if feedback_display:
                        feedback_display = feedback_display[:-3]  # Remove trailing " • "
                        st.markdown(f'<div class="public-rating-stats">{feedback_display}</div>', unsafe_allow_html=True)
                    
                    st.markdown(f'<div class="public-rating-stats">Total: {public_stats["like"]} 👍 | {public_stats["mid"]} 😐 | {public_stats["dislike"]} 👎</div>', unsafe_allow_html=True)
                    
                    # Display recent reviews
                    if 'Review' in recent_feedback.columns:  # Check if the 'Review' column exists
                        reviews_to_show = recent_feedback[recent_feedback['Review'].notna() & (recent_feedback['Review'] != "")]
                        if not reviews_to_show.empty:
                            st.markdown('<div class="recent-reviews" style="margin-top: 10px;">', unsafe_allow_html=True)
                            for _, fb in reviews_to_show.iterrows():
                                emoji = "👍" if fb['Feedback'] == 'like' else "😐" if fb['Feedback'] == 'mid' else "👎"
                                st.markdown(f'<div style="font-style: italic; margin-bottom: 5px;">{fb["Username"]} {emoji}: "{fb["Review"]}"</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="public-rating-stats">No ratings yet - be the first!</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Predicted Score", f"{row['avg_score']:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Feedback section
            with cols[3]:
                feedback_df = load_feedback()
                existing_feedback = feedback_df[
                    (feedback_df['Album Name'] == row['Album Name']) & 
                    (feedback_df['Artist'] == row['Artist'])
                ]
                
                if not existing_feedback.empty:
                    feedback = existing_feedback.iloc[0]['Feedback']
                    review_text = existing_feedback.iloc[0].get('Review', '')
                    
                    if feedback == 'like':
                        st.markdown('👍 Mike liked it')
                    elif feedback == 'mid':
                        st.markdown('😐 Mike thought it was mid')
                    elif feedback == 'dislike':
                        st.markdown('👎 Mike didn\'t like it')
                    
                    # Display Mike's review if it exists
                    if review_text and not pd.isna(review_text):
                        st.markdown(f'<div style="font-style: italic; margin-top: 5px;">"{review_text}"</div>', unsafe_allow_html=True)
                else:
                    st.markdown('😶 Mike hasn\'t listened/rated this album.')
                
            st.markdown('</div>', unsafe_allow_html=True)

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
    
    # Create tabs for different functions
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Add Missing Album Artwork", 
        "Fix Album Covers with Wrong Image", 
        "Fix Spotify Links", 
        "Nuke Albums", 
        "Manage Anonymous Reviews",
        "Data Backup & Restore"
    ])
    
    with tab1:
        st.subheader("Manage Missing Album Artwork")
        
        # Load the current album covers data and predictions data
        album_covers_df = load_album_covers()
        predictions_data = load_predictions()
        
        if predictions_data is None:
            st.error("Could not load prediction data. Please check the predictions folder.")
            return
        
        df, _ = predictions_data
        all_albums_df = df[['Artist', 'Album Name']].drop_duplicates()
        
        # Identify albums missing artwork
        merged_df = all_albums_df.merge(
            album_covers_df,
            left_on=['Artist', 'Album Name'],
            right_on=['Artist', 'Album Name'],
            how='left'
        )
        
        missing_artwork = merged_df[merged_df['Album Art'].isna()].copy()
        
        # Show statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Albums", len(all_albums_df))
        with col2:
            st.metric("Missing Artwork", len(missing_artwork))
        
        if len(missing_artwork) == 0:
            st.success("All albums have artwork! 🎉")
        else:
            # Search functionality
            st.subheader("Search Albums")
            search_query = st.text_input("Search by artist or album name:", key="artwork_search")
            
            # Quick search buttons
            quick_search_terms = ['Live', 'Deluxe', 'Reissue']
            st.write("Quick searches:")
            cols = st.columns(len(quick_search_terms))
            for i, term in enumerate(quick_search_terms):
                with cols[i]:
                    if st.button(term, key=f"quick_search_{term}"):
                        search_query = term
            
            # Filter albums based on search
            if search_query:
                filtered_missing = missing_artwork[
                    missing_artwork['Artist'].str.contains(search_query, case=False) |
                    missing_artwork['Album Name'].str.contains(search_query, case=False)
                ]
            else:
                filtered_missing = missing_artwork
            
            # Display filtered results
            st.subheader(f"Albums Missing Artwork ({len(filtered_missing)})")
            
            for _, row in filtered_missing.iterrows():
                with st.container():
                    st.markdown("---")
                    cols = st.columns([2, 3, 1])
                    
                    with cols[0]:
                        st.write(f"**Artist:** {row['Artist']}")
                        st.write(f"**Album:** {row['Album Name']}")
                    
                    with cols[1]:
                        url = st.text_input(
                            "Image URL:",
                            key=f"url_{row['Artist']}_{row['Album Name']}"
                        )
                        st.caption("Right-click image in Google and copy image address")
                    
                    with cols[2]:
                        if url:
                            try:
                                st.image(url, width=100)
                                if st.button("Save", key=f"save_{row['Artist']}_{row['Album Name']}"):
                                    # Add new row to album_covers_df
                                    new_row = {
                                        'Artist': row['Artist'],
                                        'Album Name': row['Album Name'],
                                        'Album Art': url
                                    }
                                    album_covers_df = pd.concat([album_covers_df, pd.DataFrame([new_row])], ignore_index=True)
                                    album_covers_df.to_csv('data/nmf_album_covers.csv', index=False)
                                    st.success("Saved!")
                                    st.cache_data.clear()
                                    st.rerun()
                            except:
                                st.error("Invalid image URL")

    with tab2:
        st.subheader("Fix Album Covers with Wrong Image")
        
        # Load the current album covers data and predictions data
        album_covers_df = load_album_covers()
        predictions_data = load_predictions()
        
        if predictions_data is None:
            st.error("Could not load prediction data. Please check the predictions folder.")
            return
        
        df, _ = predictions_data
        all_albums_df = df[['Artist', 'Album Name']].drop_duplicates()
        
        # Merge with album covers to get all albums with covers
        merged_df = all_albums_df.merge(
            album_covers_df,
            left_on=['Artist', 'Album Name'],
            right_on=['Artist', 'Album Name'],
            how='inner'
        )
        
        albums_with_covers = merged_df.copy()
        
        # Show statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Albums", len(all_albums_df))
        with col2:
            st.metric("Albums with Artwork", len(albums_with_covers))
        
        if len(albums_with_covers) == 0:
            st.warning("No albums with artwork found.")
        else:
            # Album selection
            st.subheader("Select an album to update its cover")
        
            selected_album_idx = st.selectbox(
                "Albums with artwork:",
                options=range(len(albums_with_covers)),
                format_func=lambda x: f"{albums_with_covers.iloc[x]['Artist']} - {albums_with_covers.iloc[x]['Album Name']}",
                key="fix_artwork_selector"
            )
            
            if selected_album_idx is not None:
                selected_album = albums_with_covers.iloc[selected_album_idx]
                artist = selected_album['Artist']
                album = selected_album['Album Name']
                current_cover_url = selected_album['Album Art']
                
                st.write(f"**Selected:** {artist} - {album}")
                
                # Display current cover
                st.subheader("Current Album Cover")
                try:
                    st.image(current_cover_url, caption=f"Current cover for {artist} - {album}", width=300)
                except Exception as e:
                    st.error(f"Failed to load current image: {e}")
                
                # Direct URL input for new cover
                st.subheader("Enter New Album Cover Image URL")
                new_url = st.text_input("New Image URL:", 
                                      value="",
                                      key="fix_artwork_url")
                
                # Helper text
                st.caption("Tip: Search for the correct album cover on Google Images, right-click on an image and select 'Copy image address'")
                
                # Preview the new URL image if provided
                if new_url:
                    st.subheader("New Album Cover Preview")
                    try:
                        st.image(new_url, caption=f"New cover for {artist} - {album}", width=300)
                    except Exception as e:
                        st.error(f"Failed to load image from URL: {e}")
                
                # Save the new URL
                if new_url and st.button("Update Album Cover", key="update_artwork"):
                    # Update the existing entry
                    album_covers_df.loc[(album_covers_df['Artist'] == artist) & 
                                      (album_covers_df['Album Name'] == album), 'Album Art'] = new_url
                    
                    # Save the updated dataframe
                    try:
                        album_covers_df.to_csv('data/nmf_album_covers.csv', index=False)
                        st.success(f"Updated album art URL for {artist} - {album}")
                        
                        # Clear cache to reflect the update
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save: {e}")

    with tab3:
        st.subheader("Fix Spotify Links")
        
        # Load the current album links data and predictions data
        album_links_df = load_album_links()
        predictions_data = load_predictions()
        
        if predictions_data is None:
            st.error("Could not load prediction data. Please check the predictions folder.")
            return
        
        df, _ = predictions_data
        all_albums_df = df[['Artist', 'Album Name']].drop_duplicates()
        
        # Rename Artist column to match album_links_df
        all_albums_df = all_albums_df.rename(columns={'Artist': 'Artist Name(s)'})
        
        # Identify albums missing Spotify links
        merged_df = all_albums_df.merge(
            album_links_df,
            left_on=['Album Name', 'Artist Name(s)'],
            right_on=['Album Name', 'Artist Name(s)'],
            how='left'
        )
        
        missing_links = merged_df[merged_df['Spotify URL'].isna()].copy()
        
        # Show statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Albums", len(all_albums_df))
        with col2:
            st.metric("Missing Spotify Links", len(missing_links))
        
        if len(missing_links) == 0:
            st.success("All albums have Spotify links! 🎉")
        else:
            # Search functionality
            st.subheader("Search Albums")
            search_query = st.text_input("Search by artist or album name:", key="spotify_search")
            
            # Quick search buttons
            quick_search_terms = ['Live', 'Deluxe', 'Reissue']
            st.write("Quick searches:")
            cols = st.columns(len(quick_search_terms))
            for i, term in enumerate(quick_search_terms):
                with cols[i]:
                    if st.button(term, key=f"spotify_quick_search_{term}"):
                        search_query = term
            
            # Filter albums based on search
            if search_query:
                filtered_missing = missing_links[
                    missing_links['Artist Name(s)'].str.contains(search_query, case=False) |
                    missing_links['Album Name'].str.contains(search_query, case=False)
                ]
            else:
                filtered_missing = missing_links
            
            # Display filtered results
            st.subheader(f"Albums Missing Spotify Links ({len(filtered_missing)})")
            
            for _, row in filtered_missing.iterrows():
                with st.container():
                    st.markdown("---")
                    cols = st.columns([2, 3, 1])
                    
                    with cols[0]:
                        st.write(f"**Artist:** {row['Artist Name(s)']}")
                        st.write(f"**Album:** {row['Album Name']}")
                    
                    with cols[1]:
                        url = st.text_input(
                            "Spotify URL:",
                            key=f"spotify_url_{row['Artist Name(s)']}_{row['Album Name']}",
                            help="Paste full Spotify URL (e.g., https://open.spotify.com/album/...)"
                        )
                        if st.button("🔍 Search on Spotify", key=f"spotify_search_{row['Artist Name(s)']}_{row['Album Name']}"):
                            search_url = f"https://open.spotify.com/search/{row['Artist Name(s)']}%20{row['Album Name']}"
                            st.markdown(f'<a href="{search_url}" target="_blank">Open Spotify Search</a>', unsafe_allow_html=True)
                    
                    with cols[2]:
                        if url:
                            if st.button("Save", key=f"save_spotify_{row['Artist Name(s)']}_{row['Album Name']}"):
                                # Create a new row for the dataframe
                                new_row = {
                                    'Album Name': row['Album Name'],
                                    'Artist Name(s)': row['Artist Name(s)'],
                                    'Spotify URL': url.replace('https://', '')  # Remove https:// prefix
                                }
                                
                                # Check if this artist/album already exists
                                existing_index = album_links_df[
                                    (album_links_df['Artist Name(s)'] == row['Artist Name(s)']) & 
                                    (album_links_df['Album Name'] == row['Album Name'])
                                ].index
                                
                                if not existing_index.empty:
                                    # Update existing entry
                                    album_links_df.loc[existing_index, 'Spotify URL'] = new_row['Spotify URL']
                                else:
                                    # Add new entry
                                    album_links_df = pd.concat([album_links_df, pd.DataFrame([new_row])], ignore_index=True)
                                
                                # Save the updated dataframe
                                try:
                                    album_links_df.to_csv('data/nmf_album_links.csv', index=False)
                                    st.success("Saved!")
                                    st.cache_data.clear()
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to save: {e}")

    with tab4:
        st.subheader("Nuke Albums")
        
        # Load the current predictions data
        predictions_data = load_predictions()
        if predictions_data is None:
            st.error("Could not load prediction data. Please check the predictions folder.")
            return
        
        df, _ = predictions_data
        
        # Load or create the nuked albums CSV
        nuked_albums_file = 'data/nuked_albums.csv'
        if os.path.exists(nuked_albums_file):
            nuked_albums_df = pd.read_csv(nuked_albums_file)
        else:
            nuked_albums_df = pd.DataFrame(columns=['Artist', 'Album Name', 'Reason'])
        
        # Initialize session state for nuked albums if not already set
        if 'nuked_albums' not in st.session_state:
            st.session_state.nuked_albums = nuked_albums_df.to_dict('records')
        
        # Show statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Albums", len(df))
        with col2:
            st.metric("Nuked Albums", len(st.session_state.nuked_albums))
        
        # Suggest albums for nuking based on keywords
        st.subheader("Suggestions for Nuking")
        keywords = ["Live", "Deluxe", "Reissue", "Anniversary"]
        suggested_albums = df[
            df['Album Name'].str.contains('|'.join(keywords), case=False, regex=True)
        ]
        
        if not suggested_albums.empty:
            st.write("Albums with keywords like 'Live', 'Deluxe', 'Reissue', or 'Anniversary':")
            for idx, row in suggested_albums.iterrows():
                # Check if the album has already been nuked
                already_nuked = any(
                    (nuked['Artist'] == row['Artist']) and (nuked['Album Name'] == row['Album Name'])
                    for nuked in st.session_state.nuked_albums
                )
                
                # Only show the button if the album hasn't been nuked
                if not already_nuked:
                    if st.button(f"Nuke {row['Artist']} - {row['Album Name']}", key=f"suggested_nuke_{idx}"):
                        # Add to nuked albums
                        new_nuke = {
                            'Artist': row['Artist'],
                            'Album Name': row['Album Name'],
                            'Reason': "Keyword match"
                        }
                        st.session_state.nuked_albums.append(new_nuke)
                        nuked_albums_df = pd.DataFrame(st.session_state.nuked_albums)
                        nuked_albums_df.to_csv(nuked_albums_file, index=False)
                        st.success(f"Nuked {row['Artist']} - {row['Album Name']}")
                        st.rerun()  # Refresh the page to update the UI
                else:
                    st.write(f"✅ {row['Artist']} - {row['Album Name']} has already been nuked.")
        else:
            st.info("No albums found with keywords like 'Live', 'Deluxe', 'Reissue', or 'Anniversary'.")
        
        # Manual nuking
        st.subheader("Manually Nuke an Album")
        all_albums = df[['Artist', 'Album Name']].drop_duplicates()
        selected_album_idx = st.selectbox(
            "Select an album to nuke:",
            options=range(len(all_albums)),
            format_func=lambda x: f"{all_albums.iloc[x]['Artist']} - {all_albums.iloc[x]['Album Name']}"
        )
        
        if selected_album_idx is not None:
            selected_album = all_albums.iloc[selected_album_idx]
            artist = selected_album['Artist']
            album = selected_album['Album Name']
            
            st.write(f"**Selected:** {artist} - {album}")
            
            # Reason for nuking
            reason = st.text_input("Reason for nuking (optional):", key=f"nuke_reason_{selected_album_idx}")
            
            if st.button("Nuke This Album", key=f"nuke_button_{selected_album_idx}"):
                # Add to nuked albums
                new_nuke = {
                    'Artist': artist,
                    'Album Name': album,
                    'Reason': reason if reason else "Manual nuke"
                }
                st.session_state.nuked_albums.append(new_nuke)
                nuked_albums_df = pd.DataFrame(st.session_state.nuked_albums)
                nuked_albums_df.to_csv(nuked_albums_file, index=False)
                st.success(f"Nuked {artist} - {album}")
                st.rerun()
        
        # Show current nuked albums
        st.subheader("Currently Nuked Albums")
        if st.session_state.nuked_albums:
            st.dataframe(pd.DataFrame(st.session_state.nuked_albums))
        else:
            st.info("No albums have been nuked yet.")

    with tab5:
        st.subheader("Manage Reviews")
        
        # Load public feedback
        public_feedback_df = load_public_feedback()
        
        # Show statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reviews", len(public_feedback_df))
        with col2:
            st.metric("Anonymous Reviews", len(public_feedback_df[public_feedback_df['Username'] == "Anonymous"]))
        with col3:
            st.metric("Mike's Reviews", len(public_feedback_df[public_feedback_df['Username'] == "Mike"]))
        with col4:
            # Count reviews with usernames similar to Mike (case insensitive)
            mike_like_reviews = public_feedback_df[
                public_feedback_df['Username'].str.lower().str.contains('mike')
            ]
            st.metric("Mike-like Reviews", len(mike_like_reviews))
        
        if public_feedback_df.empty:
            st.info("No reviews found!")
        else:
            # Add filter options
            filter_options = ["All Reviews", "Anonymous Reviews Only", "Mike's Reviews Only", 
                            "Mike-like Reviews", "Other Users' Reviews"]
            filter_choice = st.radio("Filter reviews:", filter_options)
            
            # Apply filter
            if filter_choice == "Anonymous Reviews Only":
                filtered_reviews = public_feedback_df[public_feedback_df['Username'] == "Anonymous"].copy()
            elif filter_choice == "Mike's Reviews Only":
                filtered_reviews = public_feedback_df[public_feedback_df['Username'] == "Mike"].copy()
            elif filter_choice == "Mike-like Reviews":
                filtered_reviews = public_feedback_df[
                    public_feedback_df['Username'].str.lower().str.contains('mike')
                ].copy()
            elif filter_choice == "Other Users' Reviews":
                filtered_reviews = public_feedback_df[
                    (~public_feedback_df['Username'].str.lower().str.contains('mike')) & 
                    (public_feedback_df['Username'] != "Anonymous")
                ].copy()
            else:
                filtered_reviews = public_feedback_df.copy()
            
            # Sort options
            sort_options = ["Newest First", "Oldest First", "Album Name", "Artist Name"]
            sort_choice = st.selectbox("Sort by:", sort_options)
            
            # Apply sorting
            if sort_choice == "Newest First":
                filtered_reviews['Timestamp'] = pd.to_datetime(filtered_reviews['Timestamp'])
                filtered_reviews = filtered_reviews.sort_values('Timestamp', ascending=False)
            elif sort_choice == "Oldest First":
                filtered_reviews['Timestamp'] = pd.to_datetime(filtered_reviews['Timestamp'])
                filtered_reviews = filtered_reviews.sort_values('Timestamp', ascending=True)
            elif sort_choice == "Album Name":
                filtered_reviews = filtered_reviews.sort_values('Album Name')
            elif sort_choice == "Artist Name":
                filtered_reviews = filtered_reviews.sort_values('Artist')
            
            # Display reviews with delete buttons
            st.subheader(f"Reviews ({len(filtered_reviews)})")
            
            for idx, row in filtered_reviews.iterrows():
                with st.container():
                    cols = st.columns([3, 1, 1])
                    
                    with cols[0]:
                        feedback_emoji = "👍" if row['Feedback'] == 'like' else "😐" if row['Feedback'] == 'mid' else "👎"
                        review_text = f"\"{row['Review']}\"" if row['Review'] and not pd.isna(row['Review']) else "No review text"
                        st.write(f"**{row['Artist']} - {row['Album Name']}** {feedback_emoji}")
                        st.write(f"User: **{row['Username']}** | Date: {row['Timestamp']}")
                        st.write(review_text)
                    
                    with cols[2]:
                        if st.button("Delete", key=f"delete_review_{idx}"):
                            # Remove this review
                            public_feedback_df = public_feedback_df.drop(idx)
                            
                            # Save the updated dataframe
                            public_feedback_df.to_csv('feedback/public_feedback.csv', index=False, quoting=1)
                            st.success(f"Deleted review for {row['Artist']} - {row['Album Name']} by {row['Username']}")
                            
                            # Clear cache and rerun
                            st.cache_data.clear()
                            st.rerun()
            
            # Add bulk delete options
            st.subheader("Bulk Delete Options")
            bulk_options = st.columns(4)
            
            with bulk_options[0]:
                if st.button("Delete All Anonymous Reviews"):
                    # Remove all anonymous reviews
                    public_feedback_df = public_feedback_df[public_feedback_df['Username'] != "Anonymous"]
                    
                    # Save the updated dataframe
                    public_feedback_df.to_csv('feedback/public_feedback.csv', index=False, quoting=1)
                    st.success(f"Deleted all anonymous reviews")
                    
                    # Clear cache and rerun
                    st.cache_data.clear()
                    st.rerun()
            
            with bulk_options[1]:
                if st.button("Delete All Mike-like Reviews"):
                    # Count before deletion
                    count_before = len(public_feedback_df)
                    
                    # Remove all reviews with usernames containing 'mike' (case insensitive)
                    public_feedback_df = public_feedback_df[
                        ~public_feedback_df['Username'].str.lower().str.contains('mike')
                    ]
                    
                    # Count after deletion
                    count_deleted = count_before - len(public_feedback_df)
                    
                    # Save the updated dataframe
                    public_feedback_df.to_csv('feedback/public_feedback.csv', index=False, quoting=1)
                    st.success(f"Deleted {count_deleted} Mike-like reviews")
                    
                    # Clear cache and rerun
                    st.cache_data.clear()
                    st.rerun()
            
            with bulk_options[2]:
                if st.button("Delete All Displayed Reviews"):
                    # Get the indices of the filtered reviews
                    indices_to_delete = filtered_reviews.index
                    
                    # Remove these reviews
                    public_feedback_df = public_feedback_df.drop(indices_to_delete)
                    
                    # Save the updated dataframe
                    public_feedback_df.to_csv('feedback/public_feedback.csv', index=False, quoting=1)
                    st.success(f"Deleted {len(indices_to_delete)} displayed reviews")
                    
                    # Clear cache and rerun
                    st.cache_data.clear()
                    st.rerun()
            
            with bulk_options[3]:
                if st.button("Delete All Reviews", key="delete_all_reviews"):
                    # Confirm deletion with a warning
                    st.warning("⚠️ This will delete ALL reviews! Are you sure?")
                    if st.button("Yes, Delete ALL Reviews", key="confirm_delete_all"):
                        # Create empty dataframe with same columns
                        empty_df = pd.DataFrame(columns=public_feedback_df.columns)
                        
                        # Save the empty dataframe
                        empty_df.to_csv('feedback/public_feedback.csv', index=False, quoting=1)
                        st.success(f"Deleted all {len(public_feedback_df)} reviews")
                        
                        # Clear cache and rerun
                        st.cache_data.clear()
                        st.rerun()

    with tab6:
        # Import the data_backup_restore module
        try:
            from data_backup_restore import data_backup_restore_tab
            # Call the function to render the tab content
            data_backup_restore_tab()
        except ImportError:
            st.error("Could not load data_backup_restore module. Make sure data_backup_restore.py is in your project directory.")
        except Exception as e:
            st.error(f"Error loading Data Backup & Restore tab: {e}")

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
    if 'playlist_origin' in df.columns and 'Artist Name(s)' in df.columns:
        liked_artists = set(
            df[df['playlist_origin'].isin(['df_liked', 'df_fav_albums'])]['Artist Name(s)']
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
    if include_nmf and 'playlist_origin' in df.columns and 'Artist Name(s)' in df.columns:
        nmf_artists = set(
            df[df['playlist_origin'] == 'df_nmf']['Artist Name(s)']
            .str.split(',').explode().str.strip()
        )
        not_liked_artists = set(
            df[df['playlist_origin'] == 'df_not_liked']['Artist Name(s)']
            .str.split(',').explode().str.strip()
        )
        G.add_nodes_from(nmf_artists, type='nmf')
        G.add_nodes_from(not_liked_artists, type='not_liked')
    
    return G

def main():
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
    
    # Navigation
    page_options = [
        "Weekly Predictions",
        "The Machine Learning Model",
        "6 Degrees of Lucy Dacus",
        "About Me"
    ]

    # Add Album Fixer only if running locally
    if IS_LOCAL:
        page_options.append("Album Fixer")

    page = st.sidebar.radio("Navigate", page_options)
    
    # Get all prediction files
    file_dates = get_all_prediction_files()
    
    if page == "Weekly Predictions":
        st.title("🎵 New Music Friday Regression Model")
        st.subheader("Personalized New Music Friday Recommendations")
        
        # Get the current date for display
        if len(file_dates) > 1:
            current_date = file_dates[st.session_state.current_archive_index][2]
            selected_file = file_dates[st.session_state.current_archive_index][0]
        else:
            # If only one file, use it
            selected_file = file_dates[0][0] if file_dates else None
            current_date = file_dates[0][2] if file_dates else "Unknown"
        
        # Load the selected predictions file
        predictions_data = load_predictions(selected_file)
        album_covers_df = load_album_covers()
        similar_artists_df = load_similar_artists()
        
        if predictions_data is None:
            st.error("Could not load prediction data. Please check the predictions folder.")
            return
        
        df, analysis_date = predictions_data
        
        # Load nuked albums
        nuked_albums_df = load_nuked_albums()
        
        # Filter out nuked albums
        if not nuked_albums_df.empty:
            df = df[~df.apply(lambda row: (
                (row['Artist'] in nuked_albums_df['Artist'].values) &
                (row['Album Name'] in nuked_albums_df['Album Name'].values)
            ), axis=1)]
        
        # Fixed the genre counting logic
        all_genres = set()
        for genres_str in df['Genres']:
            if isinstance(genres_str, str):
                genres_list = [g.strip() for g in genres_str.split(',')]
                all_genres.update(genres_list)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("New Releases", len(df))
        with col2:
            st.metric("Genres Analyzed", len(all_genres))
        with col3:
            st.metric("Release Week", current_date)
        
        st.subheader("🏆 Top Album Predictions")
        
        # Filter out genres that contain numbers or have more than 2 words
        filtered_genres = []
        for genre in all_genres:
            # Skip if it contains any digits
            if any(char.isdigit() for char in genre):
                continue
                
            # Count words (treating hyphenated words as separate)
            # Replace hyphens with spaces first, then count words
            modified_genre = genre.replace('-', ' ')
            word_count = len(modified_genre.split())
            
            # Only include if it has 2 or fewer words
            if word_count <= 2:
                filtered_genres.append(genre)

        genres = st.multiselect(
            "Filter by Genre",
            options=sorted(filtered_genres),
            default=[]
        )


        
        if genres:
            filtered_data = df[
                df['Genres'].apply(lambda x: any(genre in x for genre in genres))
            ]
        else:
            filtered_data = df
        
        filtered_data = filtered_data.sort_values('avg_score', ascending=False)
        display_album_predictions(filtered_data, album_covers_df, similar_artists_df)
        
        # Archive navigation at the bottom of the page
        if len(file_dates) > 1:
            st.markdown("---")
            st.markdown("### Browse Other Release Weeks")
            
            # Create a container for the archive navigation
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Current release week display
                    st.markdown(f"**Current Release Week:** {current_date}")
                
                with col2:
                    # Archive navigation buttons
                    cols = st.columns(2)
                    with cols[0]:
                        if st.session_state.current_archive_index < len(file_dates) - 1:
                            if st.button("← Older", key="older_button"):
                                st.session_state.current_archive_index += 1
                                st.rerun()
                    with cols[1]:
                        if st.session_state.current_archive_index > 0:
                            if st.button("Newer →", key="newer_button"):
                                st.session_state.current_archive_index -= 1
                                st.rerun()
                
                # Small link to view all archives
                if st.button("View All Archives", key="view_all_archives"):
                    st.session_state.show_all_archives = True
                
                # Show all archives if requested
                if st.session_state.get("show_all_archives", False):
                    st.markdown("### All Available Archives")
                    for i, (_, _, date_str) in enumerate(file_dates):
                        if st.button(date_str, key=f"archive_{i}"):
                            st.session_state.current_archive_index = i
                            st.session_state.show_all_archives = False
                            st.rerun()
                    
                    if st.button("Hide Archives", key="hide_archives"):
                        st.session_state.show_all_archives = False
                        st.rerun()
    
    elif page == "Album Fixer":
        album_fixer_page()
    
    elif page == "The Machine Learning Model":
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
    
    elif page == "About Me":
        about_me_page()
    
    elif page == "6 Degrees of Lucy Dacus":
        # Load the liked similar artists dataset
        df_liked_similar = load_liked_similar()
        
        # Use the latest predictions for the graph
        latest_file = file_dates[0][0] if file_dates else None
        predictions_data = load_predictions(latest_file)
        
        if predictions_data is None:
            st.error("Could not load prediction data. Please check the predictions folder.")
            return
            
        df, _ = predictions_data
        
        # Load the artist network graph (G)
        G = build_graph(df, df_liked_similar, include_nmf=True)
        
        dacus_game_page(G)

if __name__ == "__main__":
    try:
        # Initialize session state for archive navigation
        if 'current_archive_index' not in st.session_state:
            st.session_state.current_archive_index = 0
        if 'show_all_archives' not in st.session_state:
            st.session_state.show_all_archives = False
            
        main()
    except Exception as e:
        st.error("An error occurred during application execution:")
        st.error(str(e))
        st.code(traceback.format_exc())

except Exception as e:
    st.error("An error occurred during application startup:")
    st.error(str(e))
    st.code(traceback.format_exc())
