import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import streamlit.components.v1 as components
import glob
import os

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
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_predictions():
    prediction_files = glob.glob('predictions/*_Album_Recommendations.csv')
    if not prediction_files:
        st.error("No prediction files found!")
        return None
    
    latest_file = max(prediction_files)
    predictions_df = pd.read_csv(latest_file)
    
    date_str = os.path.basename(latest_file).split('_')[0]
    analysis_date = datetime.strptime(date_str, '%m-%d-%y').strftime('%Y-%m-%d')
    
    return predictions_df, analysis_date

@st.cache_data
def load_album_covers():
    return pd.read_csv('data/nmf_album_covers.csv')

@st.cache_data
def load_similar_artists():
    return pd.read_csv('data/nmf_similar_artists.csv')

def load_training_data():
    df = pd.read_csv('data/df_cleaned_pre_standardized.csv')
    return df[df['playlist_origin'] != 'df_nmf'].copy()

# [Previous code remains the same until the display_album_predictions function's CSS]

def display_album_predictions(filtered_data, album_covers_df, similar_artists_df):
    # Enhanced CSS for better visual hierarchy and layout
    st.markdown("""
        <style>
        .large-text {
            font-size: 1.2rem !important;
            line-height: 1.6 !important;
            margin: 8px 0 !important;
            text-align: left !important;
        }
        .album-title {
            font-size: 1.4rem !important;
            margin-bottom: 16px !important;
            font-weight: 600 !important;
            text-align: left !important;
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

    try:
        album_links_df = pd.read_csv('data/nmf_album_links.csv')
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
    
    filtered_albums = merged_data[
        (merged_data['Album Art'].notna()) | (merged_data['avg_score'] >= 40)
    ]
    
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
                st.markdown(f'<div class="album-title">{row["Artist"]} - {row["Album Name"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="large-text"><strong>Genre:</strong> {row["Genres"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="large-text"><strong>Label:</strong> {row["Label"]}</div>', unsafe_allow_html=True)
                
                similar_artists = similar_artists_df[
                    similar_artists_df['Artist'] == row['Artist']
                ]
                
                if not similar_artists.empty:
                    similar_list = similar_artists.iloc[0]['Similar Artists']
                    st.markdown(f'<div class="large-text"><strong>Similar Artists:</strong> {similar_list}</div>', unsafe_allow_html=True)
                
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
            
            with cols[3]:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Confidence", f"{row['confidence_score']:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.sidebar.title("About This Project")
    st.sidebar.write("""
    ### Tech Stack
    - 🤖 Machine Learning: RandomForest & XGBoost
    - 📊 Data Processing: Pandas & NumPy
    - 🎨 Visualization: Plotly & Streamlit
    - 🎵 Data Source: Spotify API
    
    ### Key Features
    - Weekly New Music Predictions
    - Advanced Artist Similarity Analysis
    - Genre-based Learning
    - Automated Label Analysis
    """)
    
    page = st.sidebar.radio(
        "Navigate",
        ["Weekly Predictions", "Notebook"]
    )
    
    predictions_data = load_predictions()
    if predictions_data is None:
        st.error("Could not load prediction data. Please check the predictions folder.")
        return
    
    df, analysis_date = predictions_data
    album_covers_df = load_album_covers()
    similar_artists_df = load_similar_artists()
    
    if page == "Weekly Predictions":
        st.title("🎵 New Music Friday Regression Model")
        st.subheader("Personalized New Music Friday Recommendations")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("New Releases", len(df))
        with col2:
            st.metric("Genres Analyzed", df['Genres'].nunique())
        with col3:
            st.metric("Analysis Date", analysis_date)
        
        st.subheader("🏆 Top Album Predictions")
        
        all_genres = set()
        for genres in df['Genres'].str.split(','):
            if isinstance(genres, list):
                all_genres.update([g.strip() for g in genres])
        
        genres = st.multiselect(
            "Filter by Genre",
            options=sorted(list(all_genres)),
            default=[]
        )
        
        artist_search = st.text_input(
            "Search by Artist",
            placeholder="Enter artist name..."
        ).strip().lower()
        
        if genres or artist_search:
            filtered_data = df.copy()
            if genres:
                filtered_data = filtered_data[
                    filtered_data['Genres'].apply(lambda x: any(genre in x for genre in genres))
                ]
            if artist_search:
                filtered_data = filtered_data[
                    filtered_data['Artist'].str.lower().str.contains(artist_search, na=False)
                ]
        else:
            filtered_data = df
        
        filtered_data = filtered_data.sort_values('avg_score', ascending=False)
        display_album_predictions(filtered_data, album_covers_df, similar_artists_df)
    
    elif page == "Notebook":
        st.title("📓 Jupyter Notebook")
        st.subheader("Embedded notebook content below:")
        
        try:
            with open('graphics/Music_Taste_Machine_Learning_Data_Prep.html', 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.markdown('<div class="notebook-content">', unsafe_allow_html=True)
            components.html(html_content, height=800, scrolling=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading notebook content: {e}")

if __name__ == "__main__":
    main()