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

# Check if App is Running Locally or on Streamlit's Servers
def is_running_on_streamlit():
    return os.getenv("STREAMLIT_SERVER_RUNNING", "False").lower() == "true"

# We no longer need to check if running locally
# IS_LOCAL = not is_running_on_streamlit()

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
        flex-wrap: wrap;
    }
    .public-rating-buttons button {
        flex: 1;
        min-width: 60px;
        padding: 8px 12px;
        font-size: 0.9rem;
    }
    .public-rating-stats {
        font-size: 0.9rem;
        color: #666;
        margin-top: 5px;
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
    /* Media query for mobile devices */
    @media (max-width: 768px) {
        .public-rating-buttons {
            justify-content: space-between;
        }
        .public-rating-buttons button {
            min-width: 60px;
        }
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
def load_similar_artists():
    """
    Load similar artists data from the CSV file.
    """
    similar_artists_file = 'data/nmf_similar_artists.csv'
    if os.path.exists(similar_artists_file):
        try:
            return pd.read_csv(similar_artists_file)
        except Exception as e:
            st.error(f"Error loading similar artists: {e}")
    return pd.DataFrame(columns=['Artist', 'Similar Artists'])

def save_feedback(album_name, artist, feedback):
    """
    Save Mike's feedback to a CSV file.
    """
    feedback_file = 'feedback/feedback.csv'
    if not os.path.exists('feedback'):
        os.makedirs('feedback')
    
    # Create a dataframe with the new feedback
    new_feedback = pd.DataFrame({
        'Album Name': [album_name],
        'Artist': [artist],
        'Feedback': [feedback]
    })
    
    # Load existing feedback if file exists
    if os.path.exists(feedback_file):
        try:
            existing_feedback = pd.read_csv(feedback_file)
            
            # Check if feedback for this album already exists
            mask = (existing_feedback['Album Name'] == album_name) & (existing_feedback['Artist'] == artist)
            if mask.any():
                # Update existing feedback
                existing_feedback.loc[mask, 'Feedback'] = feedback
                existing_feedback.to_csv(feedback_file, index=False)
            else:
                # Append new feedback
                pd.concat([existing_feedback, new_feedback]).to_csv(feedback_file, index=False)
        except Exception as e:
            st.error(f"Error updating feedback: {e}")
            # If there's an error, just overwrite the file
            new_feedback.to_csv(feedback_file, index=False)
    else:
        # Create new feedback file
        new_feedback.to_csv(feedback_file, index=False)

def load_feedback():
    """
    Load Mike's feedback from a CSV file.
    """
    feedback_file = 'feedback/feedback.csv'
    if os.path.exists(feedback_file):
        try:
            return pd.read_csv(feedback_file)
        except Exception as e:
            st.error(f"Error loading feedback: {e}")
    return pd.DataFrame(columns=['Album Name', 'Artist', 'Feedback'])

def save_public_feedback(album_name, artist, feedback, username="Anonymous"):
    """
    Save public feedback to a CSV file.
    """
    feedback_file = 'feedback/public_feedback.csv'
    if not os.path.exists('feedback'):
        os.makedirs('feedback')
    
    # Create a dataframe with the new feedback
    new_feedback = pd.DataFrame({
        'Album Name': [album_name],
        'Artist': [artist],
        'Feedback': [feedback],
        'Username': [username],
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })
    
    # Load existing feedback if file exists
    if os.path.exists(feedback_file):
        try:
            existing_feedback = pd.read_csv(feedback_file)
            
            # Append new feedback (we allow multiple feedbacks from the same user)
            pd.concat([existing_feedback, new_feedback]).to_csv(feedback_file, index=False)
        except Exception as e:
            st.error(f"Error updating public feedback: {e}")
            # If there's an error, just overwrite the file
            new_feedback.to_csv(feedback_file, index=False)
    else:
        # Create new feedback file
        new_feedback.to_csv(feedback_file, index=False)

def load_public_feedback():
    """
    Load public feedback from a CSV file.
    """
    feedback_file = 'feedback/public_feedback.csv'
    if os.path.exists(feedback_file):
        try:
            return pd.read_csv(feedback_file)
        except Exception as e:
            st.error(f"Error loading public feedback: {e}")
    return pd.DataFrame(columns=['Album Name', 'Artist', 'Feedback', 'Username', 'Timestamp'])

def get_public_feedback_stats(album_name, artist):
    """
    Get statistics for public feedback on a specific album.
    """
    public_feedback_df = load_public_feedback()
    
    # Filter for this album
    album_feedback = public_feedback_df[
        (public_feedback_df['Album Name'] == album_name) & 
        (public_feedback_df['Artist'] == artist)
    ]
    
    # Count different feedback types
    like_count = len(album_feedback[album_feedback['Feedback'] == 'like'])
    mid_count = len(album_feedback[album_feedback['Feedback'] == 'mid'])
    dislike_count = len(album_feedback[album_feedback['Feedback'] == 'dislike'])
    total_count = len(album_feedback)
    
    return {
        'like': like_count,
        'mid': mid_count,
        'dislike': dislike_count,
        'total': total_count
    }

def get_recent_public_feedback(album_name, artist, limit=3):
    """
    Get the most recent public feedback for a specific album.
    """
    public_feedback_df = load_public_feedback()
    
    # Filter for this album
    album_feedback = public_feedback_df[
        (public_feedback_df['Album Name'] == album_name) & 
        (public_feedback_df['Artist'] == artist)
    ]
    
    # Sort by timestamp (newest first) and limit
    if 'Timestamp' in album_feedback.columns:
        album_feedback = album_feedback.sort_values('Timestamp', ascending=False)
    
    return album_feedback.head(limit)

@st.cache_data
def load_album_covers():
    """
    Load album cover URLs from the CSV file.
    """
    covers_file = 'data/album_covers.csv'
    if os.path.exists(covers_file):
        try:
            return pd.read_csv(covers_file)
        except Exception as e:
            st.error(f"Error loading album covers: {e}")
    return pd.DataFrame(columns=['Artist', 'Album Name', 'Album Art'])

def save_album_cover(artist, album_name, cover_url):
    """
    Save an album cover URL to the CSV file.
    """
    covers_file = 'data/album_covers.csv'
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Create a dataframe with the new cover
    new_cover = pd.DataFrame({
        'Artist': [artist],
        'Album Name': [album_name],
        'Album Art': [cover_url]
    })
    
    # Load existing covers if file exists
    if os.path.exists(covers_file):
        try:
            existing_covers = pd.read_csv(covers_file)
            
            # Check if cover for this album already exists
            mask = (existing_covers['Artist'] == artist) & (existing_covers['Album Name'] == album_name)
            if mask.any():
                # Update existing cover
                existing_covers.loc[mask, 'Album Art'] = cover_url
                existing_covers.to_csv(covers_file, index=False)
            else:
                # Append new cover
                pd.concat([existing_covers, new_cover]).to_csv(covers_file, index=False)
        except Exception as e:
            st.error(f"Error updating album covers: {e}")
            # If there's an error, just overwrite the file
            new_cover.to_csv(covers_file, index=False)
    else:
        # Create new covers file
        new_cover.to_csv(covers_file, index=False)

@st.cache_data
def load_album_links():
    """
    Load album Spotify links from the CSV file.
    """
    links_file = 'data/album_links.csv'
    if os.path.exists(links_file):
        try:
            return pd.read_csv(links_file)
        except Exception as e:
            st.error(f"Error loading album links: {e}")
    return pd.DataFrame(columns=['Artist Name(s)', 'Album Name', 'Spotify URL'])

# Function to convert Spotify links to open.spotify format
def convert_spotify_link(url):
    """
    Convert Spotify links to open.spotify format if they start with http
    """
    if url and isinstance(url, str):
        # Check if the URL starts with http and contains spotify.com
        if url.startswith('http') and 'spotify.com' in url:
            # If it's already in the format https://open.spotify.com, just return it
            if 'open.spotify.com' in url:
                return url
                
            # Otherwise, try to extract the path and create a new open.spotify.com URL
            parts = url.split('spotify.com')
            if len(parts) > 1:
                return f"open.spotify.com{parts[1]}"
            
            # Replace http(s)://spotify.com with open.spotify.com
            return url.replace('https://spotify.com', 'open.spotify.com').replace('http://spotify.com', 'open.spotify.com')
    
    # Return the original URL if no conversion was needed or possible
    return url

def save_album_link(artist, album_name, spotify_url):
    """
    Save an album Spotify link to the CSV file.
    """
    links_file = 'data/album_links.csv'
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Create a dataframe with the new link
    new_link = pd.DataFrame({
        'Artist Name(s)': [artist],
        'Album Name': [album_name],
        'Spotify URL': [spotify_url]
    })
    
    # Load existing links if file exists
    if os.path.exists(links_file):
        try:
            existing_links = pd.read_csv(links_file)
            
            # Check if link for this album already exists
            mask = (existing_links['Artist Name(s)'] == artist) & (existing_links['Album Name'] == album_name)
            if mask.any():
                # Update existing link
                existing_links.loc[mask, 'Spotify URL'] = spotify_url
                existing_links.to_csv(links_file, index=False)
            else:
                # Append new link
                pd.concat([existing_links, new_link]).to_csv(links_file, index=False)
        except Exception as e:
            st.error(f"Error updating album links: {e}")
            # If there's an error, just overwrite the file
            new_link.to_csv(links_file, index=False)
    else:
        # Create new links file
        new_link.to_csv(links_file, index=False)

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
    
    # Display each album
    for idx, (_, row) in enumerate(merged_data.iterrows()):
        st.markdown(f'<div class="album-container">', unsafe_allow_html=True)
        
        # Create columns for layout
        cols = st.columns([1, 2, 1, 1])
        
        # Album artwork
        with cols[0]:
            if 'Album Art' in row and pd.notna(row['Album Art']):
                st.image(row['Album Art'], width=200)
            else:
                st.markdown("*Artwork not available*")
        
        # Album details
        with cols[1]:
            st.markdown(f"### {row['Album Name']}")
            st.markdown(f"**Artist:** {row['Artist']}")
            
            # Display similar artists if available
            similar_artists = similar_artists_df[
                similar_artists_df['Artist'] == row['Artist']
            ]
            
            if not similar_artists.empty:
                similar_list = similar_artists.iloc[0]['Similar Artists']
                st.markdown(f'<div class="similar-artists">Similar to: {similar_list}</div>', unsafe_allow_html=True)
            
            # Display label if available - with check to avoid KeyError
            if 'label' in row and pd.notna(row['label']):
                st.markdown(f"**Label:** {row['label']}")
            
            # Display genres if available - with check to avoid KeyError
            if 'genres' in row and pd.notna(row['genres']):
                st.markdown(f"**Genres:** {row['genres']}")
            
            # Spotify link if available
            if 'Spotify URL' in row and pd.notna(row['Spotify URL']):
                spotify_url = row['Spotify URL']
                # Convert the Spotify URL if needed
                spotify_url = convert_spotify_link(spotify_url)
                st.markdown(f'''
                    <a href="https://{spotify_url}" target="_blank" class="spotify-button">
                        ▶ Play on Spotify
                    </a>
                ''', unsafe_allow_html=True)
            
                # Public rating section with username input
                st.markdown('<div style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 8px;">', unsafe_allow_html=True)
                st.markdown('<div style="font-weight: 600; margin-bottom: 8px;">Mike wants to know what you think!</div>', unsafe_allow_html=True)
                
                # Username input
                username = st.text_input("Your name (optional):", key=f"username_input_{idx}", value="")
                username = username.strip() if username else "Anonymous"
                
                # Check if this is Mike with the identifier
                is_mike = username == "Mike S"
                display_username = "Mike" if is_mike else username
                
                # Rating buttons
                st.markdown('<div class="public-rating-buttons">', unsafe_allow_html=True)
                public_rating_cols = st.columns(3)
                with public_rating_cols[0]:
                    if st.button('👍', key=f"public_like_{idx}"):
                        if is_mike:
                            # Use Mike's private feedback system
                            save_feedback(row['Album Name'], row['Artist'], 'like')
                        else:
                            # Use public feedback system
                            save_public_feedback(row['Album Name'], row['Artist'], 'like', display_username)
                        st.rerun()
                with public_rating_cols[1]:
                    if st.button('😐', key=f"public_mid_{idx}"):
                        if is_mike:
                            # Use Mike's private feedback system
                            save_feedback(row['Album Name'], row['Artist'], 'mid')
                        else:
                            # Use public feedback system
                            save_public_feedback(row['Album Name'], row['Artist'], 'mid', display_username)
                        st.rerun()
                with public_rating_cols[2]:
                    if st.button('👎', key=f"public_dislike_{idx}"):
                        if is_mike:
                            # Use Mike's private feedback system
                            save_feedback(row['Album Name'], row['Artist'], 'dislike')
                        else:
                            # Use public feedback system
                            save_public_feedback(row['Album Name'], row['Artist'], 'dislike', display_username)
                        st.rerun()
                
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
                if feedback == 'like':
                    st.markdown('👍 Mike liked it')
                elif feedback == 'mid':
                    st.markdown('😐 Mike thought it was mid')
                elif feedback == 'dislike':
                    st.markdown('👎 Mike didn\'t like it')
            else:
                st.markdown('😶 Mike hasn\'t listened/rated this album.')
            
            # We no longer need Mike's feedback buttons since we're using the name parsing approach
        
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
    st.write("🔗 Connect with me on [LinkedIn](https://www.linkedin.com/in/mike-strusz/)")
    
    st.image("graphics/mike.jpeg", width=400)
    st.caption("Me on the Milwaukee Riverwalk, wearing one of my 50+ bowties.")

def manage_album_covers():
    st.title("🖼️ Album Cover Manager")
    
    # Load the current album covers data and predictions data
    album_covers_df = load_album_covers()
    predictions_data = load_predictions()
    
    if predictions_data is None:
        st.error("Could not load prediction data. Please check the predictions folder.")
        return
    
    # Extract album information
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
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Albums", len(all_albums_df))
    with col2:
        st.metric("Albums with Artwork", len(all_albums_df) - len(missing_artwork))
    with col3:
        st.metric("Albums Missing Artwork", len(missing_artwork))
    
    if len(missing_artwork) == 0:
        st.success("All albums have artwork! 🎉")
        return
    
    # Select an album to add artwork for
    selected_album = st.selectbox(
        "Select an album to add artwork:",
        missing_artwork.apply(lambda x: f"{x['Artist']} - {x['Album Name']}", axis=1).tolist()
    )
    
    if selected_album:
        artist, album_name = selected_album.split(" - ", 1)
        
        st.write(f"Adding artwork for: **{album_name}** by **{artist}**")
        
        # Input for artwork URL
        artwork_url = st.text_input("Paste the album artwork URL:")
        
        if artwork_url:
            # Preview the image
            try:
                response = requests.get(artwork_url)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption=f"{album_name} by {artist}", width=300)
                
                # Save button
                if st.button("Save Artwork"):
                    save_album_cover(artist, album_name, artwork_url)
                    st.success(f"Artwork saved for {album_name}!")
                    st.cache_data.clear()
                    sleep(1)
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading image: {e}")

def manage_spotify_links():
    st.title("🎵 Spotify Link Manager")
    st.subheader("Manage Missing Spotify Links")
    
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
        left_on=['Artist Name(s)', 'Album Name'],
        right_on=['Artist Name(s)', 'Album Name'],
        how='left'
    )
    
    missing_links = merged_df[merged_df['Spotify URL'].isna()].copy()
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Albums", len(all_albums_df))
    with col2:
        st.metric("Albums with Spotify Links", len(all_albums_df) - len(missing_links))
    with col3:
        st.metric("Albums Missing Spotify Links", len(missing_links))
    
    if len(missing_links) == 0:
        st.success("All albums have Spotify links! 🎉")
        return
    
    # Select an album to add Spotify link for
    selected_album = st.selectbox(
        "Select an album to add Spotify link:",
        missing_links.apply(lambda x: f"{x['Artist Name(s)']} - {x['Album Name']}", axis=1).tolist()
    )
    
    if selected_album:
        artist, album_name = selected_album.split(" - ", 1)
        
        st.write(f"Adding Spotify link for: **{album_name}** by **{artist}**")
        
        # Input for Spotify URL
        spotify_url = st.text_input(
            "Paste the Spotify URL (will be automatically converted if needed):",
            key=f"{artist}_{album_name}_spotify_url",
            value=st.session_state.get(f"{artist}_{album_name}_spotify_url", ""))
        
        if spotify_url:
            # Convert the Spotify URL if needed
            converted_url = convert_spotify_link(spotify_url)
            
            # If the URL was converted, show the conversion
            if converted_url != spotify_url:
                st.info(f"URL converted to: {converted_url}")
                spotify_url = converted_url
            
            # Preview the Spotify link
            st.markdown(f'''
                <a href="https://{spotify_url}" target="_blank" class="spotify-button">
                    ▶ Preview on Spotify
                </a>
            ''', unsafe_allow_html=True)
            
            # Save button
            if st.button("Save Spotify Link"):
                save_album_link(artist, album_name, spotify_url)
                st.success(f"Spotify link saved for {album_name}!")
                st.cache_data.clear()
                sleep(1)
                st.rerun()

def fix_existing_spotify_links():
    st.subheader("Fix Existing Spotify Links")
    
    # Load the current album links data
    album_links_df = load_album_links()
    
    if len(album_links_df) == 0:
        st.warning("No album links data found.")
        return
    
    # Select an album to fix Spotify link for
    albums_with_links = album_links_df.apply(lambda x: f"{x['Artist Name(s)']} - {x['Album Name']}", axis=1).tolist()
    selected_album = st.selectbox(
        "Select an album to fix Spotify link:",
        albums_with_links,
        key="fix_spotify_selectbox"
    )
    
    if selected_album:
        artist, album_name = selected_album.split(" - ", 1)
        
        # Get current Spotify link
        current_link = album_links_df[
            (album_links_df['Artist Name(s)'] == artist) & 
            (album_links_df['Album Name'] == album_name)
        ]['Spotify URL'].values[0]
        
        st.write(f"Fixing Spotify link for: **{album_name}** by **{artist}**")
        
        # Display current link
        st.markdown(f"Current link: `{current_link}`")
        st.markdown(f'''
            <a href="https://{current_link}" target="_blank" class="spotify-button">
                ▶ Current Spotify Link
            </a>
        ''', unsafe_allow_html=True)
        
        # Input for new Spotify URL
        new_spotify_url = st.text_input(
            "Paste the new Spotify URL (will be automatically converted if needed):", 
            key="new_spotify_url")
        
        if new_spotify_url:
            # Convert the Spotify URL if needed
            converted_url = convert_spotify_link(new_spotify_url)
            
            # If the URL was converted, show the conversion
            if converted_url != new_spotify_url:
                st.info(f"URL converted to: {converted_url}")
                new_spotify_url = converted_url
            
            # Preview the new Spotify link
            st.markdown(f'''
                <a href="https://{new_spotify_url}" target="_blank" class="spotify-button">
                    ▶ Preview New Link
                </a>
            ''', unsafe_allow_html=True)
            
            # Save button
            if st.button("Update Spotify Link"):
                save_album_link(artist, album_name, new_spotify_url)
                st.success(f"Spotify link updated for {album_name}!")
                st.cache_data.clear()
                sleep(1)
                st.rerun()

def lucy_dacus_page():
    st.title("🎵 6 Degrees of Lucy Dacus")
    st.subheader("Artist Connection Explorer")
    
    st.write("""
    Select an artist to see how closely they're connected to Lucy Dacus!
    The **Dacus number** is the number of connections between the artist and Lucy Dacus.
    """)
    
    # Load similar artists data
    similar_artists_df = load_similar_artists()
    
    if similar_artists_df.empty:
        st.error("Similar artists data not found. Please check the data folder.")
        return
    
    # Create a graph of artist connections
    G = nx.Graph()
    
    # Add all artists as nodes
    all_artists = set(similar_artists_df['Artist'].unique())
    for row in similar_artists_df.iterrows():
        artist = row[1]['Artist']
        similar_str = row[1]['Similar Artists']
        
        if pd.notna(similar_str) and similar_str:
            similar_list = [s.strip() for s in similar_str.split(',')]
            
            # Add edges between artist and similar artists
            for similar in similar_list:
                if similar:  # Skip empty strings
                    G.add_edge(artist, similar)
                    all_artists.add(similar)
    
    # Add all artists as nodes
    G.add_nodes_from(all_artists)
    
    # Create a sorted list of artists for the selectbox
    sorted_artists = sorted(list(all_artists))
    
    # Artist selection
    selected_artist = st.selectbox("Select an artist:", sorted_artists)
    
    if selected_artist:
        # Calculate Dacus number
        dacus_number, path = calculate_dacus_number(G, selected_artist)
        
        # Display results
        if dacus_number is not None:
            st.markdown(f"### {selected_artist} has a Dacus number of {dacus_number}")
            st.write(f"**Path to Lucy Dacus:** {' → '.join(path)}")
            
            # Visualize the path
            visualize_path(G, path)
        else:
            st.error("No path found. This artist might not be connected to Lucy Dacus in our network.")

def calculate_dacus_number(G, artist_name):
    """
    Calculate the Dacus number (shortest path length to Lucy Dacus) for an artist.
    Returns the Dacus number and the path.
    """
    try:
        # If the artist is Lucy Dacus, the Dacus number is 0
        if artist_name == "Lucy Dacus":
            return 0, ["Lucy Dacus"]
        
        # Find the shortest path to Lucy Dacus
        path = nx.shortest_path(G, source=artist_name, target="Lucy Dacus")
        return len(path) - 1, path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, None

def visualize_path(G, path):
    """
    Visualize the artist network and highlight the path to Lucy Dacus.
    """
    # Create a subgraph with the path and immediate neighbors
    subgraph_nodes = set(path)
    for node in path:
        subgraph_nodes.update(G.neighbors(node))
    
    subgraph = G.subgraph(subgraph_nodes)
    
    # Create a Plotly figure
    pos = nx.spring_layout(subgraph, seed=42)
    
    # Create edges
    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        # Color nodes in the path differently
        if node in path:
            if node == "Lucy Dacus":
                node_color.append('#FF5733')  # Lucy Dacus in orange
            elif node == path[0]:
                node_color.append('#33FF57')  # Selected artist in green
            else:
                node_color.append('#3357FF')  # Path nodes in blue
        else:
            node_color.append('#CCCCCC')  # Other nodes in gray
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            color=node_color,
            size=15,
            line_width=2))
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=f'Path from {path[0]} to Lucy Dacus',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    # Initialize session state for feedback updates
    if 'feedback_updated' not in st.session_state:
        st.session_state.feedback_updated = False

    # Check if feedback was updated and clear cache if needed
    if st.session_state.feedback_updated:
        st.cache_data.clear()
        st.session_state.feedback_updated = False

    # For archive navigation
    if 'current_archive_index' not in st.session_state:
        st.session_state.current_archive_index = 0
    
    # Create sidebar navigation
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
    
    # Define page options
    page_options = [
        "Weekly Predictions",
        "The Machine Learning Model",
        "6 Degrees of Lucy Dacus",
        "About Me"
    ]
    
    # Add Album Cover Manager and Spotify Link Manager only if running locally
    # We no longer check IS_LOCAL, but we'll keep these options available
    page_options.append("Album Cover Manager")
    page_options.append("Spotify Link Manager")
    
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
        
        genres = st.multiselect(
            "Filter by Genre",
            options=sorted(list(all_genres)),
            default=[]
        )
        
        # Filter by genre if selected
        filtered_data = df
        if genres:
            filtered_data = df[df['Genres'].apply(
                lambda x: any(genre in str(x).split(',') for genre in genres) if pd.notna(x) else False
            )]
        
        # Display album predictions
        display_album_predictions(filtered_data, album_covers_df, similar_artists_df)
        
        # Archives navigation at the bottom of the page
        st.markdown("---")
        st.subheader("Browse Other Weeks")
        
        # Create a row for navigation buttons
        cols = st.columns([1, 1, 1])
        
        with cols[0]:
            if st.session_state.current_archive_index < len(file_dates) - 1:
                if st.button("← Older Week"):
                    st.session_state.current_archive_index += 1
                    st.rerun()
        
        with cols[1]:
            st.write(f"**Current: {current_date}**")
        
        with cols[2]:
            if st.session_state.current_archive_index > 0:
                if st.button("Newer Week →"):
                    st.session_state.current_archive_index -= 1
                    st.rerun()
    
    elif page == "Album Cover Manager":
        manage_album_covers()
    
    elif page == "Spotify Link Manager":
        manage_spotify_links()
    
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
            st.error(f"Error loading notebook: {e}")
            st.write("Please make sure the notebook HTML file exists in the graphics folder.")
    
    elif page == "6 Degrees of Lucy Dacus":
        lucy_dacus_page()
    
    elif page == "About Me":
        about_me_page()

# Initialize session state for archive viewing
if 'show_all_archives' not in st.session_state:
    st.session_state.show_all_archives = False
    
if __name__ == "__main__":
    main()
