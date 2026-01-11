import streamlit as st
from verify_data import verify_app_data

# Run data verification at the start
verify_app_data()
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

# Use this flag to control feedback buttons
IS_LOCAL = not is_running_on_streamlit()

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
    
    # Standardize column names for the new model
    if 'Album' in predictions_df.columns:
        predictions_df['Album Name'] = predictions_df['Album']
    
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

def save_gut_score(album_name, artist, score, file_path):
    """
    CLEAN VERSION: Only saves when there's an actual score (0-100)
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
        
        # Save to WEEKLY file
        weekly_file = f'feedback/{date_str}_gut_scores.csv'
        os.makedirs('feedback', exist_ok=True)
        
        # Create new entry
        new_entry = pd.DataFrame([{
            'Album': album_name,
            'Artist': artist,
            'gut_score': float(score),  # Ensure float
            'gut_score_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source_file': source_filename,
            'predicted_score': None  # Will be filled from predictions file
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
        
        # Save to MASTER file
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
        
        st.success(f"✅ Gut score {score} saved successfully!")
        
        # Optional: Also trigger Thursday processing immediately
        st.info("💡 This score will be added to training data on Thursday")
        
    except Exception as e:
        st.error(f"❌ Error saving gut score: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# The display_album_predictions function
def display_album_predictions(filtered_data, album_covers_df, similar_artists_df, current_file_path=None):
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
            # Standardize columns if not already done by load_album_links
            if 'Artist Name(s)' in album_links_df.columns:
                album_links_df = album_links_df.rename(columns={'Artist Name(s)': 'Artist'})
                
            merged_data = merged_data.merge(
                album_links_df[['Album Name', 'Artist', 'Spotify URL']],
                on=['Album Name', 'Artist'],
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
                
                # Display new model columns
                if 'Track_Count' in row:
                    st.markdown(f'<div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;"><strong>Tracks:</strong> {int(row["Track_Count"])}</div>', unsafe_allow_html=True)
                
                if 'Match_Type' in row and 'Matched_To' in row:
                    match_info = f"{row['Match_Type'].replace('_', ' ').title()} ({row['Matched_To']})"
                    st.markdown(f'<div class="large-text" style="font-size: 1.2rem; line-height: 1.6; margin: 8px 0;"><strong>Match:</strong> {match_info}</div>', unsafe_allow_html=True)

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
            
            with cols[2]:
                # Metric container for predicted score
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                score = row['Predicted_Score'] if 'Predicted_Score' in row else row['avg_score']
                st.metric("Predicted Score", f"{score:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Gut Score section - now displayed in its own container below predicted score
                st.markdown('<div class="gut-score-container">', unsafe_allow_html=True)
                st.markdown("**🎯 Gut Score (0-100)**")
                
                # Get current gut score if it exists (check both column names)
                current_gut = 0
                gut_col = None
                
                # Check which column name exists
                if 'gut_score' in row:
                    gut_col = 'gut_score'
                elif 'Gut_Score' in row:
                    gut_col = 'Gut_Score'
                    
                if gut_col and pd.notna(row[gut_col]):
                    current_gut = float(row[gut_col])
                    # Display current gut score
                    st.markdown(f"**Current:** {int(current_gut)}")
                else:
                    st.markdown("**Current:** Not rated")
                
                # Use a columns layout for the score input and save button
                score_cols = st.columns([3, 2])
                with score_cols[0]:
                    new_score = st.number_input(
                        "New Score", 
                        0, 
                        100, 
                        int(current_gut), 
                        key=f"gut_input_{row['Album Name']}_{row['Artist']}",
                        label_visibility="collapsed"
                    )
                with score_cols[1]:
                    if st.button("💾 Save", key=f"gut_save_{row['Album Name']}_{row['Artist']}"):
                        save_gut_score(row['Album Name'], row['Artist'], new_score, current_file_path)
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            
            with cols[3]:
                # Removed feedback section - now only showing if Mike has already given feedback
                # Load existing feedback to display what Mike thought
                try:
                    # Try to load feedback from the CSV
                    if os.path.exists('feedback/feedback.csv'):
                        feedback_df = pd.read_csv('feedback/feedback.csv', quoting=1)
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
                    else:
                        st.markdown('😶 No feedback yet')
                except Exception as e:
                    st.markdown('😶 Error loading feedback')
            
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
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Missing Artwork", 
        "Spotify Links", 
        "Fix Any Current Album",
        "Nuke Albums"
    ])

    with tab1:
        st.subheader("Add Missing Artwork (Current Week ONLY)")
        
        # Find which current albums are missing artwork
        missing = []
        for _, curr_row in current_albums.iterrows():
            artist = curr_row['Artist']
            album = curr_row['Album Name']
            
            # Check if this album has artwork
            has_artwork = not album_covers_df[
                (album_covers_df['Artist'] == artist) & 
                (album_covers_df['Album Name'] == album) &
                (pd.notna(album_covers_df['Album Art'])) &
                (album_covers_df['Album Art'] != '')
            ].empty
            
            if not has_artwork:
                missing.append({'Artist': artist, 'Album Name': album})
        
        missing_df = pd.DataFrame(missing)
        
        st.metric("Missing Artwork This Week", len(missing_df))
        
        if len(missing_df) > 0:
            for idx, row in missing_df.iterrows():
                with st.container():
                    st.markdown(f"**{row['Artist']}** - *{row['Album Name']}*")
                    
                    # Create three columns: album info, URL input, preview, save button
                    col1, col2, col3, col4 = st.columns([2, 3, 2, 1])
                    
                    with col1:
                        # Show current state (empty)
                        st.write("Current: ❌ No artwork")
                    
                    with col2:
                        url = st.text_input(
                            "Image URL", 
                            key=f"url_{row['Artist']}_{row['Album Name']}_{idx}".replace(' ', '_'),
                            placeholder="https://example.com/image.jpg"
                        )
                    
                    with col3:
                        if url:
                            try:
                                # Try to display preview
                                st.image(url, width=100, caption="Preview")
                            except Exception as e:
                                st.error(f"❌ Can't load image: {str(e)[:50]}...")
                        else:
                            st.write("Enter URL for preview")
                    
                    with col4:
                        if st.button("💾 Save", key=f"save_{row['Artist']}_{row['Album Name']}_{idx}".replace(' ', '_')):
                            if url and url.strip():
                                # Check if already exists in full covers DF
                                mask = (album_covers_df['Artist'] == row['Artist']) & (album_covers_df['Album Name'] == row['Album Name'])
                                
                                if not album_covers_df[mask].empty:
                                    # Update existing
                                    album_covers_df.loc[mask, 'Album Art'] = url
                                else:
                                    # Add new
                                    new_row = pd.DataFrame([{
                                        'Artist': row['Artist'], 
                                        'Album Name': row['Album Name'], 
                                        'Album Art': url
                                    }])
                                    album_covers_df = pd.concat([album_covers_df, new_row], ignore_index=True)
                                
                                # Save to CSV
                                album_covers_df.to_csv('data/nmf_album_covers.csv', index=False)
                                
                                # Clear cache to force reload
                                st.cache_data.clear()
                                
                                st.success(f"✅ Artwork saved!")
                                sleep(0.5)  # Small delay to show success
                                st.rerun()
                            else:
                                st.error("Please enter a valid URL")
                    
                    st.markdown("---")
        else:
            st.success("🎉 All current week albums have artwork!")

    with tab2:
        st.subheader("Spotify Links (Current Week ONLY)")
        
        # Find which current albums are missing links
        missing_links = []
        for _, curr_row in current_albums.iterrows():
            artist = curr_row['Artist']
            album = curr_row['Album Name']
            
            # Check if this album has a link
            has_link = not album_links_df[
                (album_links_df['Artist'] == artist) & 
                (album_links_df['Album Name'] == album) &
                (pd.notna(album_links_df['Spotify URL'])) &
                (album_links_df['Spotify URL'] != '')
            ].empty
            
            if not has_link:
                missing_links.append({'Artist': artist, 'Album Name': album})
        
        missing_links_df = pd.DataFrame(missing_links)
        
        st.metric("Missing Links This Week", len(missing_links_df))
        
        if len(missing_links_df) > 0:
            for idx, row in missing_links_df.iterrows():
                with st.container():
                    st.markdown(f"**{row['Artist']}** - *{row['Album Name']}*")
                    
                    col1, col2, col3 = st.columns([2, 3, 1])
                    
                    with col1:
                        st.write("Current: ❌ No link")
                    
                    with col2:
                        s_url = st.text_input(
                            "Spotify URL", 
                            key=f"surl_{row['Artist']}_{row['Album Name']}_{idx}".replace(' ', '_'),
                            placeholder="open.spotify.com/album/..."
                        )
                        
                        if s_url:
                            # Show preview of what will be saved
                            clean_url = s_url.replace('https://', '').replace('http://', '').strip()
                            st.caption(f"Will save as: `{clean_url[:40]}...`")
                    
                    with col3:
                        if st.button("💾 Save", key=f"ssave_{row['Artist']}_{row['Album Name']}_{idx}".replace(' ', '_')):
                            if s_url and s_url.strip():
                                # Clean the URL
                                clean_url = s_url.replace('https://', '').replace('http://', '').strip()
                                
                                # Check if already exists in full links DF
                                mask = (album_links_df['Artist'] == row['Artist']) & (album_links_df['Album Name'] == row['Album Name'])
                                
                                if not album_links_df[mask].empty:
                                    # Update existing
                                    album_links_df.loc[mask, 'Spotify URL'] = clean_url
                                else:
                                    # Add new
                                    new_row = pd.DataFrame([{
                                        'Artist': row['Artist'], 
                                        'Album Name': row['Album Name'], 
                                        'Spotify URL': clean_url
                                    }])
                                    album_links_df = pd.concat([album_links_df, new_row], ignore_index=True)
                                
                                # Save to CSV
                                album_links_df.to_csv('data/nmf_album_links.csv', index=False)
                                
                                # Clear cache to force reload
                                st.cache_data.clear()
                                
                                st.success(f"✅ Link saved!")
                                sleep(0.5)  # Small delay to show success
                                st.rerun()
                            else:
                                st.error("Please enter a valid URL")
                    
                    st.markdown("---")
        else:
            st.success("🎉 All current week albums have Spotify links!")

    with tab3:
        st.subheader("Fix Any Current Album")
        st.write("Update artwork or links for any album on the front page.")
        
        # Create list for selectbox
        album_list = []
        for idx, row in current_albums.iterrows():
            album_list.append(f"{row['Artist']} - {row['Album Name']}")
        
        if not album_list:
            st.warning("No current albums found!")
            return
            
        selected_album = st.selectbox("Select Album to Fix", album_list)
        
        if selected_album:
            # Parse artist and album from selection
            artist, album = selected_album.split(" - ", 1)
            
            # Get current state
            curr_art = album_covers_df[
                (album_covers_df['Artist'] == artist) & 
                (album_covers_df['Album Name'] == album)
            ]
            curr_link = album_links_df[
                (album_links_df['Artist'] == artist) & 
                (album_links_df['Album Name'] == album)
            ]
            
            # Create session state for preview URLs
            if f'preview_art_{artist}_{album}' not in st.session_state:
                st.session_state[f'preview_art_{artist}_{album}'] = ""
            if f'preview_link_{artist}_{album}' not in st.session_state:
                st.session_state[f'preview_link_{artist}_{album}'] = ""
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**🎨 Artwork for {artist}**")
                
                # Current artwork
                if not curr_art.empty and pd.notna(curr_art.iloc[0]['Album Art']) and curr_art.iloc[0]['Album Art'].strip():
                    current_art_url = curr_art.iloc[0]['Album Art']
                    st.image(current_art_url, width=200, caption="Current Artwork")
                    st.caption(f"URL: {current_art_url[:50]}...")
                else:
                    st.warning("No artwork currently set")
                
                # New artwork input with live preview
                new_art = st.text_input(
                    "New Artwork URL", 
                    value=st.session_state[f'preview_art_{artist}_{album}'],
                    key=f"fix_art_{artist}_{album}".replace(' ', '_'),
                    placeholder="https://example.com/image.jpg"
                )
                
                # Update preview
                st.session_state[f'preview_art_{artist}_{album}'] = new_art
                
                # Preview new artwork
                if new_art and new_art.strip():
                    try:
                        st.image(new_art, width=200, caption="Preview of New Artwork")
                    except Exception as e:
                        st.error(f"❌ Can't preview image: {str(e)[:50]}")
                
                if st.button("💾 Update Artwork", key=f"update_art_{artist}_{album}".replace(' ', '_')):
                    if new_art and new_art.strip():
                        # Update or add artwork
                        mask = (album_covers_df['Artist'] == artist) & (album_covers_df['Album Name'] == album)
                        
                        if not curr_art.empty:
                            album_covers_df.loc[mask, 'Album Art'] = new_art
                        else:
                            new_row = pd.DataFrame([{
                                'Artist': artist, 
                                'Album Name': album, 
                                'Album Art': new_art
                            }])
                            album_covers_df = pd.concat([album_covers_df, new_row], ignore_index=True)
                        
                        # Save to CSV
                        album_covers_df.to_csv('data/nmf_album_covers.csv', index=False)
                        
                        # Clear cache to force reload
                        st.cache_data.clear()
                        
                        # Clear preview
                        st.session_state[f'preview_art_{artist}_{album}'] = ""
                        
                        st.success("✅ Artwork Updated!")
                        sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Please enter a valid URL")
            
            with col2:
                st.markdown(f"**🔗 Spotify Link for {artist}**")
                
                # Current link
                if not curr_link.empty and pd.notna(curr_link.iloc[0]['Spotify URL']) and curr_link.iloc[0]['Spotify URL'].strip():
                    current_url = curr_link.iloc[0]['Spotify URL']
                    full_url = f"https://{current_url}"
                    st.markdown(f"**Current Link:**")
                    st.code(current_url)
                    st.markdown(f"[🌐 Open in Spotify]({full_url})", unsafe_allow_html=True)
                else:
                    st.warning("No link currently set")
                
                # New link input
                new_link = st.text_input(
                    "New Spotify URL", 
                    value=st.session_state[f'preview_link_{artist}_{album}'],
                    key=f"fix_link_{artist}_{album}".replace(' ', '_'),
                    placeholder="open.spotify.com/album/..."
                )
                
                # Update preview
                st.session_state[f'preview_link_{artist}_{album}'] = new_link
                
                # Show what will be saved
                if new_link and new_link.strip():
                    clean_link = new_link.replace('https://', '').replace('http://', '').strip()
                    st.markdown(f"**Will save as:**")
                    st.code(clean_link)
                
                if st.button("💾 Update Link", key=f"update_link_{artist}_{album}".replace(' ', '_')):
                    if new_link and new_link.strip():
                        # Clean the URL
                        clean_link = new_link.replace('https://', '').replace('http://', '').strip()
                        
                        # Update or add link
                        mask = (album_links_df['Artist'] == artist) & (album_links_df['Album Name'] == album)
                        
                        if not curr_link.empty:
                            album_links_df.loc[mask, 'Spotify URL'] = clean_link
                        else:
                            new_row = pd.DataFrame([{
                                'Artist': artist, 
                                'Album Name': album, 
                                'Spotify URL': clean_link
                            }])
                            album_links_df = pd.concat([album_links_df, new_row], ignore_index=True)
                        
                        # Save to CSV
                        album_links_df.to_csv('data/nmf_album_links.csv', index=False)
                        
                        # Clear cache to force reload
                        st.cache_data.clear()
                        
                        # Clear preview
                        st.session_state[f'preview_link_{artist}_{album}'] = ""
                        
                        st.success("✅ Link Updated!")
                        sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Please enter a valid URL")

    with tab4:
        st.subheader("Nuke Albums (Current Week ONLY)")
        nuke_file = 'data/nuked_albums.csv'
        nuked_df = pd.read_csv(nuke_file) if os.path.exists(nuke_file) else pd.DataFrame(columns=['Artist', 'Album Name', 'Reason'])
        
        # Create list for selectbox
        album_list = []
        for idx, row in current_albums.iterrows():
            album_list.append(f"{row['Artist']} - {row['Album Name']}")
        
        if not album_list:
            st.warning("No current albums found!")
            return
            
        selected_album = st.selectbox("Select Album to Nuke", album_list)
        reason = st.text_input("Reason for nuking", placeholder="e.g., Not my genre, Already heard, Bad reviews")
        
        if st.button("💣 Nuke It!", key="nuke_button"):
            if selected_album and reason:
                # Parse artist and album from selection
                artist, album = selected_album.split(" - ", 1)
                
                # Check if already nuked
                already_nuked = nuked_df[
                    (nuked_df['Artist'] == artist) & 
                    (nuked_df['Album Name'] == album)
                ]
                
                if not already_nuked.empty:
                    st.warning(f"{artist} - {album} is already nuked!")
                else:
                    new_nuke = pd.DataFrame([{
                        'Artist': artist, 
                        'Album Name': album, 
                        'Reason': reason
                    }])
                    nuked_df = pd.concat([nuked_df, new_nuke], ignore_index=True)
                    nuked_df.to_csv(nuke_file, index=False)
                    
                    # Clear cache to force reload
                    st.cache_data.clear()
                    
                    st.success(f"💣 Nuked {artist} - {album}! Reason: {reason}")
                    sleep(0.5)
                    st.rerun()
            else:
                st.error("Please select an album and provide a reason")
        
        # Show currently nuked albums from this week
        current_nuked = []
        for _, row in current_albums.iterrows():
            artist = row['Artist']
            album = row['Album Name']
            if not nuked_df[(nuked_df['Artist'] == artist) & (nuked_df['Album Name'] == album)].empty:
                current_nuked.append({'Artist': artist, 'Album Name': album})
        
        if current_nuked:
            st.markdown("---")
            st.subheader("📋 Currently Nuked This Week")
            for item in current_nuked:
                reason_text = nuked_df[
                    (nuked_df['Artist'] == item['Artist']) & 
                    (nuked_df['Album Name'] == item['Album Name'])
                ].iloc[0]['Reason']
                st.markdown(f"❌ **{item['Artist']}** - *{item['Album Name']}*")
                st.caption(f"Reason: {reason_text}")

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
        
        sort_col = 'Predicted_Score' if 'Predicted_Score' in filtered_data.columns else 'avg_score'
        filtered_data = filtered_data.sort_values(sort_col, ascending=False)
        display_album_predictions(filtered_data, album_covers_df, similar_artists_df, selected_file)
        
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