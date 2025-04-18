import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import streamlit.components.v1 as components
import glob
import os
import requests
from time import sleep
from typing import Dict
import networkx as nx 
import numpy as np

# Check if App is Running Locally or on Streamlit's Servers
def is_running_on_streamlit():
    return os.getenv("STREAMLIT_SERVER_RUNNING", "False").lower() == "true"

# Use this flag to control feedback buttons
IS_LOCAL = not is_running_on_streamlit()

def extract_date_from_filename(file_path):
    """Extract date from filename in format MM-DD-YY"""
    filename = os.path.basename(file_path)
    date_str = filename.split('_')[0]
    try:
        date_obj = datetime.strptime(date_str, '%m-%d-%y')
        return date_obj.strftime('%B %d, %Y')
    except ValueError:
        return "Unknown Date"

def show_placeholder_art():
    """Display placeholder album art"""
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; 
                  height: 300px; background-color: #f0f0f0; border-radius: 10px;">
            <span style="font-size: 48px;">🎵</span>
        </div>
        """, 
        unsafe_allow_html=True
    )

@st.cache_data
def load_nuked_albums():
    """
    Load the list of nuked albums from the CSV file.
    """
    nuked_albums_file = 'data/nuked_albums.csv'
    if os.path.exists(nuked_albums_file):
        return pd.read_csv(nuked_albums_file)
    return pd.DataFrame(columns=['Artist', 'Album Name', 'Reason'])

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
def load_predictions(file_path=None):
    """Load predictions with built-in cover art and Spotify links"""
    if file_path is None:
        prediction_files = glob.glob('predictions/*_Album_Recommendations.csv')
        file_path = max(prediction_files) if prediction_files else None
    
    if not file_path:
        st.error("No prediction files found!")
        return None

    df = pd.read_csv(file_path)
    
    # Validate required columns
    required_cols = ['Artist', 'Album Name', 'avg_score', 
                    'Album Art URL', 'Spotify URL']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Prediction file missing columns: {', '.join(missing)}")
        return None
    
    # Clean data
    df['Album Art URL'] = df['Album Art URL'].replace('nan', np.nan)
    df['Spotify URL'] = df['Spotify URL'].replace('nan', np.nan)
    
    return df, extract_date_from_filename(file_path)

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
def load_album_covers():
    try:
        return pd.read_csv('data/nmf_album_covers.csv')
    except Exception as e:
        st.error(f"Error loading album covers data: {e}")
        return pd.DataFrame(columns=['Artist', 'Album Name', 'Album Art'])

@st.cache_data
def load_album_links():
    try:
        return pd.read_csv('data/nmf_album_links.csv')
    except Exception as e:
        st.error(f"Error loading album links data: {e}")
        return pd.DataFrame(columns=['Album Name', 'Artist Name(s)', 'Spotify URL'])

@st.cache_data
def load_similar_artists():
    try:
        return pd.read_csv('data/nmf_similar_artists.csv')
    except Exception as e:
        st.error(f"Error loading similar artists data: {e}")
        return pd.DataFrame(columns=['Artist', 'Similar Artists'])

@st.cache_data
def load_liked_similar():
    """
    Load the dataset of similar artists for liked artists.
    """
    try:
        return pd.read_csv('data/liked_artists_only_similar.csv')
    except Exception as e:
        st.error(f"Error loading liked similar artists data: {e}")
        return pd.DataFrame(columns=['Artist', 'Similar Artists'])

def load_training_data():
    df = pd.read_csv('data/df_cleaned_pre_standardized.csv')
    return df[df['playlist_origin'] != 'df_nmf'].copy()

# Add feedback functions
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
    feedback_file = 'feedback/public_feedback.csv'
    if not os.path.exists('feedback'):
        os.makedirs('feedback')
    
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
            # Use proper quoting and escape characters when reading
            existing_feedback = pd.read_csv(feedback_file, quoting=1)  # QUOTE_ALL
            # Combine with new feedback
            combined_feedback = pd.concat([existing_feedback, new_feedback], ignore_index=True)
        except Exception as e:
            st.warning(f"Error reading public feedback: {e}. Creating new file.")
            # If reading fails, start fresh with just the new feedback
            combined_feedback = new_feedback
    else:
        combined_feedback = new_feedback
    
    # Save with proper quoting to handle commas in fields
    combined_feedback.to_csv(feedback_file, index=False, quoting=1)  # QUOTE_ALL
    
    # Clear cache after saving feedback
    st.cache_data.clear()

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

def validate_prediction_file(df):
    """Check data quality before display"""
    errors = []
    
    # Check URL formats
    art_urls = df['Album Art URL'].dropna()
    spotify_urls = df['Spotify URL'].dropna()
    
    if not all(art_urls.str.startswith(('http://', 'https://'))):
        errors.append("Some cover art URLs are malformed")
    
    if not all(spotify_urls.str.contains('spotify.com')):
        errors.append("Some Spotify URLs are invalid")
    
    return errors if errors else None

def display_album_predictions(filtered_data):
    """Render album cards using consolidated data"""
    for idx, row in filtered_data.iterrows():
        with st.container():
            cols = st.columns([2, 4, 1, 1])
            
            # Column 1: Album Art
            with cols[0]:
                if pd.notna(row['Album Art URL']):
                    st.image(row['Album Art URL'], width=300)
                else:
                    show_placeholder_art()
            
            # Column 2: Album Info
            with cols[1]:
                st.markdown(f"**{row['Artist']} - {row['Album Name']}**")
                
                # Spotify link
                if pd.notna(row['Spotify URL']):
                    spotify_url = row['Spotify URL']
                    if not spotify_url.startswith('http'):
                        spotify_url = f'https://{spotify_url}'
                    st.markdown(f"[▶ Play on Spotify]({spotify_url})")
                
                # Genre and label information
                st.markdown(f"**Genre:** {row['Genres']}")
                st.markdown(f"**Label:** {row['Label']}")
                
                # Public rating section with username input
                st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                st.markdown('<div style="font-weight: 600; margin-bottom: 8px;">Mike wants to know what you think!</div>', unsafe_allow_html=True)

                # Username input
                username = st.text_input("Your name (optional):", key=f"username_input_{idx}", value="")
                username = username.strip() if username else "Anonymous"

                # Create a unique key using album name and artist
                unique_key = f"{row['Album Name']}_{row['Artist']}"

                # Add review input field with empty value to prevent persistence
                review = st.text_area("Mini review (optional):", 
                                    value="", 
                                    key=f"review_input_{unique_key}", 
                                    max_chars=200, 
                                    height=80)

                # Create fixed-width columns for buttons
                button_cols = st.columns(3)

                # Like Button
                with button_cols[0]:
                    if st.button('👍 Like', key=f"public_like_{unique_key}", use_container_width=True):
                        if username == "Mike S":
                            save_feedback(row['Album Name'], row['Artist'], 'like', review)
                            username = "Mike"
                        else:
                            save_public_feedback(row['Album Name'], row['Artist'], 'like', username, review)
                        st.rerun()

                # Mid Button
                with button_cols[1]:
                    if st.button('😐 Mid', key=f"public_mid_{unique_key}", use_container_width=True):
                        if username == "Mike S":
                            save_feedback(row['Album Name'], row['Artist'], 'mid', review)
                            username = "Mike"
                        else:
                            save_public_feedback(row['Album Name'], row['Artist'], 'mid', username, review)
                        st.rerun()

                # Dislike Button
                with button_cols[2]:
                    if st.button('👎 Dislike', key=f"public_dislike_{unique_key}", use_container_width=True):
                        if username == "Mike S":
                            save_feedback(row['Album Name'], row['Artist'], 'dislike', review)
                            username = "Mike"
                        else:
                            save_public_feedback(row['Album Name'], row['Artist'], 'dislike', username, review)
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
                    if 'Review' in recent_feedback.columns:
                        reviews_to_show = recent_feedback[recent_feedback['Review'].notna() & (recent_feedback['Review'] != "")]
                        if not reviews_to_show.empty:
                            st.markdown('<div class="recent-reviews" style="margin-top: 10px;">', unsafe_allow_html=True)
                            for _, fb in reviews_to_show.iterrows():
                                emoji = "👍" if fb['Feedback'] == 'like' else "😐" if fb['Feedback'] == 'mid' else "👎"
                                st.markdown(f'<div style="font-style: italic; margin-bottom: 5px;">{fb["Username"]} {emoji}: "{fb["Review"]}"</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="public-rating-stats">No ratings yet - be the first!</div>', unsafe_allow_html=True)
            
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
                    
                    if review_text and not pd.isna(review_text):
                        st.markdown(f'<div style="font-style: italic; margin-top: 5px;">"{review_text}"</div>', unsafe_allow_html=True)
                else:
                    st.markdown('😶 Mike hasn\'t listened/rated this album.')

def album_fixer_page():
    st.title("🛠️ Album Fixer")
    
    # Load the current prediction file
    predictions_data = load_predictions()
    if predictions_data is None:
        st.error("Could not load prediction data.")
        return
    
    df, _ = predictions_data
    
    with st.expander("⚡ Quick Fixes"):
        # Add direct editing of URLs
        selected_idx = st.selectbox("Select album to fix", df.index,
                                  format_func=lambda x: f"{df.loc[x]['Artist']} - {df.loc[x]['Album Name']}")
        
        col1, col2 = st.columns(2)
        with col1:
            new_art = st.text_input("New Cover URL", df.loc[selected_idx, 'Album Art URL'])
        with col2:
            new_spotify = st.text_input("New Spotify URL", df.loc[selected_idx, 'Spotify URL'])
        
        if st.button("Save Changes"):
            df.loc[selected_idx, 'Album Art URL'] = new_art
            df.loc[selected_idx, 'Spotify URL'] = new_spotify
            df.to_csv(glob.glob('predictions/*_Album_Recommendations.csv')[0], index=False)
            st.success("Updated!")
            st.cache_data.clear()
            st.rerun()

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
        if predictions_data is None:
            st.error("Could not load prediction data. Please check the predictions folder.")
            return
        
        df, analysis_date = predictions_data
        
        # Validate data
        if errors := validate_prediction_file(df):
            st.error("Data issues found: " + "; ".join(errors))
        
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
        display_album_predictions(filtered_data)
        
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
    # Initialize session state for archive navigation
    if 'current_archive_index' not in st.session_state:
        st.session_state.current_archive_index = 0
    if 'show_all_archives' not in st.session_state:
        st.session_state.show_all_archives = False
        
    main()