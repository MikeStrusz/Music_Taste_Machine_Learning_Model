import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import streamlit.components.v1 as components
import glob
import os
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
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_predictions():
    """
    Load the predictions data and ensure required columns are present.
    """
    prediction_files = glob.glob('predictions/*_Album_Recommendations.csv')
    if not prediction_files:
        st.error("No prediction files found!")
        return None
    
    latest_file = max(prediction_files)
    predictions_df = pd.read_csv(latest_file)
    
    # Ensure 'playlist_origin' column exists (silently add if missing)
    if 'playlist_origin' not in predictions_df.columns:
        predictions_df['playlist_origin'] = 'unknown'  # Default value
    
    # Ensure 'Artist Name(s)' column exists (silently add if missing)
    if 'Artist Name(s)' not in predictions_df.columns:
        predictions_df['Artist Name(s)'] = 'Unknown Artist'  # Default value
    
    date_str = os.path.basename(latest_file).split('_')[0]
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
def save_feedback(album_name, artist, feedback):
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
            # Use proper quoting and escape characters when reading
            existing_feedback = pd.read_csv(feedback_file, quoting=1)  # QUOTE_ALL
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
                return pd.DataFrame(columns=['Album Name', 'Artist', 'Feedback'])
    return pd.DataFrame(columns=['Album Name', 'Artist', 'Feedback'])

if 'feedback_updated' not in st.session_state:
    st.session_state.feedback_updated = False

# Then in save_feedback:
st.session_state.feedback_updated = True

# And check at the beginning of your app:
if st.session_state.feedback_updated:
    st.cache_data.clear()
    st.session_state.feedback_updated = False

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
    
    # Removed the filtering condition to show all albums
    filtered_albums = merged_data
    
    for idx, row in filtered_albums.iterrows():
        with st.container():
            st.markdown('<div class="album-container">', unsafe_allow_html=True)
            cols = st.columns([2, 4, 1, 1])  # Define cols here
            
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
                
                # Only show feedback buttons if running locally
                if not is_running_on_streamlit():
                    if st.button('👍', key=f"like_{idx}"):
                        save_feedback(row['Album Name'], row['Artist'], 'like')
                        st.experimental_rerun()
                    if st.button('😐', key=f"mid_{idx}"):
                        save_feedback(row['Album Name'], row['Artist'], 'mid')
                        st.experimental_rerun()
                    if st.button('👎', key=f"dislike_{idx}"):
                        save_feedback(row['Album Name'], row['Artist'], 'dislike')
                        st.experimental_rerun()
            
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
        return
        
    # Album selection
    st.subheader("Select an album to update")
    
    selected_album_idx = st.selectbox(
        "Albums missing artwork:",
        options=range(len(missing_artwork)),
        format_func=lambda x: f"{missing_artwork.iloc[x]['Artist']} - {missing_artwork.iloc[x]['Album Name']}"
    )
    
    if selected_album_idx is not None:
        selected_album = missing_artwork.iloc[selected_album_idx]
        artist = selected_album['Artist']
        album = selected_album['Album Name']
        
        st.write(f"**Selected:** {artist} - {album}")
        
        # Direct URL input
        st.subheader("Enter Album Cover Image URL")
        direct_url = st.text_input("Image URL:", 
                                  value=st.session_state.get(f"{artist}_{album}_url", ""))
        
        # Helper text
        st.caption("Tip: Search for the album cover on Google Images, right-click on an image and select 'Copy image address'")
        
        # Preview the URL image if provided
        if direct_url:
            try:
                st.image(direct_url, caption=f"{artist} - {album}", width=300)
            except Exception as e:
                st.error(f"Failed to load image from URL: {e}")
        
        # Save the direct URL
        if direct_url and st.button("Save URL"):
            # Create a new row for the dataframe
            new_row = {
                'Artist': artist,
                'Album Name': album,
                'Album Art': direct_url
            }
            
            # Check if this artist/album already exists
            existing = album_covers_df[(album_covers_df['Artist'] == artist) & 
                                     (album_covers_df['Album Name'] == album)]
            
            if not existing.empty:
                # Update existing entry
                album_covers_df.loc[(album_covers_df['Artist'] == artist) & 
                                  (album_covers_df['Album Name'] == album), 'Album Art'] = direct_url
            else:
                # Add new entry
                album_covers_df = pd.concat([album_covers_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save the updated dataframe
            try:
                album_covers_df.to_csv('data/nmf_album_covers.csv', index=False)
                st.success(f"Saved album art URL for {artist} - {album}")
                
                # Clear cache to reflect the update
                st.cache_data.clear()
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")

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
        return
        
    # Album selection
    st.subheader("Select an album to update")
    
    selected_album_idx = st.selectbox(
        "Albums missing Spotify links:",
        options=range(len(missing_links)),
        format_func=lambda x: f"{missing_links.iloc[x]['Artist Name(s)']} - {missing_links.iloc[x]['Album Name']}"
    )
    
    if selected_album_idx is not None:
        selected_album = missing_links.iloc[selected_album_idx]
        artist = selected_album['Artist Name(s)']
        album = selected_album['Album Name']
        
        st.write(f"**Selected:** {artist} - {album}")
        
        # Direct URL input
        st.subheader("Enter Spotify URL")
        direct_url = st.text_input("Spotify URL:", 
                                  value=st.session_state.get(f"{artist}_{album}_spotify_url", ""))
        
        # Helper text
        st.caption("Tip: Search for the album on Spotify, click 'Share', then 'Copy Link'. Paste the URL here without the 'https://' prefix.")
        
        # Format the URL if needed
        if direct_url and direct_url.startswith('https://'):
            direct_url = direct_url.replace('https://', '')
            st.info("Removed 'https://' prefix from URL")
        
        # Save the direct URL
        if direct_url and st.button("Save URL"):
            # Create a new row for the dataframe
            new_row = {
                'Album Name': album,
                'Artist Name(s)': artist,
                'Spotify URL': direct_url
            }
            
            # Check if this artist/album already exists
            existing = album_links_df[(album_links_df['Artist Name(s)'] == artist) & 
                                    (album_links_df['Album Name'] == album)]
            
            if not existing.empty:
                # Update existing entry
                album_links_df.loc[(album_links_df['Artist Name(s)'] == artist) & 
                                 (album_links_df['Album Name'] == album), 'Spotify URL'] = direct_url
            else:
                # Add new entry
                album_links_df = pd.concat([album_links_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save the updated dataframe
            try:
                album_links_df.to_csv('data/nmf_album_links.csv', index=False)
                st.success(f"Saved Spotify URL for {artist} - {album}")
                
                # Clear cache to reflect the update
                st.cache_data.clear()
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")

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
        st.experimental_rerun()
    
    # Navigation
    page_options = [
        "Weekly Predictions",
        "The Machine Learning Model",
        "6 Degrees of Lucy Dacus",
        "About Me"
    ]

    # Add Album Cover Manager and Spotify Link Manager only if running locally
    if not is_running_on_streamlit():
        page_options.append("Album Cover Manager")
        page_options.append("Spotify Link Manager")

    page = st.sidebar.radio("Navigate", page_options)
    
    # Load datasets
    predictions_data = load_predictions()
    album_covers_df = load_album_covers()
    similar_artists_df = load_similar_artists()
    
    if predictions_data is None:
        st.error("Could not load prediction data. Please check the predictions folder.")
        return
    
    df, analysis_date = predictions_data
    
    # Load the liked similar artists dataset
    df_liked_similar = load_liked_similar()
    
    # Load the artist network graph (G)
    G = build_graph(df, df_liked_similar, include_nmf=True)
    
    if page == "Weekly Predictions":
        st.title("🎵 New Music Friday Regression Model")
        st.subheader("Personalized New Music Friday Recommendations")
        
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
            formatted_date = datetime.strptime(analysis_date, '%Y-%m-%d').strftime('%B %d, %Y')
            st.metric("Analysis Date", formatted_date)
        
        st.subheader("🏆 Top Album Predictions")
        
        genres = st.multiselect(
            "Filter by Genre",
            options=sorted(list(all_genres)),
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
            st.error(f"Error loading notebook content: {e}")
    
    elif page == "About Me":
        about_me_page()
    
    elif page == "6 Degrees of Lucy Dacus":
        dacus_game_page(G)

if __name__ == "__main__":
    main()
