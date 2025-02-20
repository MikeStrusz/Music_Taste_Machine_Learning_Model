import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import glob
import os

st.set_page_config(
    page_title="New Music Friday Regression Model",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_predictions():
    # Get the most recent predictions file
    prediction_files = glob.glob('predictions/*_Album_Recommendations.csv')
    if not prediction_files:
        st.error("No prediction files found!")
        return None

    latest_file = max(prediction_files)
    predictions_df = pd.read_csv(latest_file)
    
    # Extract date from filename
    date_str = os.path.basename(latest_file).split('_')[0]
    analysis_date = datetime.strptime(date_str, '%m-%d-%y').strftime('%Y-%m-%d')
    
    return predictions_df, analysis_date

def load_training_data():
    df = pd.read_csv('data/df_cleaned_pre_standardized.csv')
    # Filter out NMF data since we only want training data
    return df[df['playlist_origin'] != 'df_nmf'].copy()

def main():
    # Sidebar with project context
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
    
    # Main navigation
    page = st.sidebar.radio(
        "Navigate",
        ["Weekly Predictions", "Technical Overview", "Music Analytics", "Explore Mike's Taste"]
    )
    
    # Load prediction data
    predictions_data = load_predictions()
    if predictions_data is None:
        st.error("Could not load prediction data. Please check the predictions folder.")
        return
    
    df, analysis_date = predictions_data
    
    if page == "Weekly Predictions":
        st.title("🎵 New Music Friday Regression Model")
        st.subheader("Personalized New Music Friday Recommendations")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("New Releases", len(df))
        with col2:
            st.metric("Genres Analyzed", df['Genres'].nunique())
        with col3:
            st.metric("Analysis Date", analysis_date)
            
        # Top predictions with interactive elements
        st.subheader("🏆 Top Album Predictions")
        
        # Create a set of all unique genres
        all_genres = set()
        for genres in df['Genres'].str.split(','):
            if isinstance(genres, list):
                all_genres.update([g.strip() for g in genres])
        
        # Genre filter
        genres = st.multiselect(
            "Filter by Genre",
            options=sorted(list(all_genres)),
            default=[]
        )
        
        # Filter data based on selected genres
        if genres:
            filtered_data = df[
                df['Genres'].apply(lambda x: any(genre in x for genre in genres))
            ]
        else:
            filtered_data = df
        
        # Artist search
        artist_search = st.text_input(
            "Search by Artist",
            placeholder="Enter artist name..."
        ).strip().lower()
        
        # Modify filtering logic to include both genre and artist filters
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

        # Sort by average score
        top_albums = filtered_data.sort_values('avg_score', ascending=False)
        
        # Display predictions in an engaging format
        for idx, row in top_albums.head(10).iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"### {row['Artist']} - {row['Album Name']}")
                    st.markdown(f"**Genre:** {row['Genres']}")
                    st.markdown(f"**Label:** {row['Label']}")
                with col2:
                    st.metric("Score", f"{row['avg_score']:.1f}")
                with col3:
                    st.metric("Confidence", f"{row['confidence_score']:.1f}")
                st.markdown("---")
        
    elif page == "Technical Overview":
        st.title("🔬 Model Architecture")
        
        st.write("""
        ### Prediction Pipeline
        This model combines multiple ML techniques to predict music preferences:
        1. Artist Similarity Network Analysis
        2. Genre Encoding & Classification
        3. Audio Feature Analysis
        4. Ensemble Prediction
        """)
        
        # Feature importance visualization
        st.subheader("🎯 Feature Impact Analysis")
        features = ['Artist Network Centrality', 'Label Analysis', 'Genre Encoding', 
                   'Popularity Metrics', 'Mood Profile', 'Energy Signature']
        importance = [0.365, 0.317, 0.160, 0.056, 0.039, 0.038]
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='#1DB954'  # Spotify green
        ))
        fig.update_layout(
            title="Feature Importance in Prediction Model",
            xaxis_title="Relative Importance",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Explore Mike's Taste":
        st.title("🎧 Mike's Music Taste Analysis")
        
        # Load training data
        training_df = load_training_data()
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Overview", "Audio Features", "Timeline Analysis"])
        
        with tab1:
            st.subheader("Music Collection Overview")
            
            # Display collection metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                loved = len(training_df[training_df['playlist_origin'] == 'df_liked'])
                st.metric("Loved Songs (100%)", loved)
            with col2:
                liked_albums = training_df[training_df['playlist_origin'] == 'df_fav_albums']['Album Name'].nunique()
                st.metric("Liked Albums (50%)", liked_albums)
            with col3:
                disliked_albums = training_df[training_df['playlist_origin'] == 'df_not_liked']['Album Name'].nunique()
                st.metric("Disliked Albums (0%)", disliked_albums)
            
            # Top genres pie chart
            st.subheader("Favorite Genres in Loved Songs")
            loved_genres = training_df[training_df['playlist_origin'] == 'df_liked']['Genres'].str.split('|', expand=True).stack()
            top_genres = loved_genres.value_counts().head(10)
            
            fig = px.pie(
                values=top_genres.values,
                names=top_genres.index,
                title="Top 10 Genres in Loved Songs"
            )
            st.plotly_chart(fig)
            
        with tab2:
            st.subheader("Audio Features Analysis")
            
            # Feature distribution by preference level
            feature_cols = ['Danceability', 'Energy', 'Valence', 'Tempo', 'mood_score', 'energy_profile']
            
            selected_feature = st.selectbox(
                "Select Audio Feature to Analyze",
                feature_cols
            )
            
            fig = go.Figure()
            for origin, name, color in [
                ('df_liked', 'Loved Songs', '#1DB954'),
                ('df_fav_albums', 'Liked Albums', '#1ED760'),
                ('df_not_liked', 'Disliked Albums', '#FF6B6B')
            ]:
                data = training_df[training_df['playlist_origin'] == origin][selected_feature]
                fig.add_trace(go.Violin(
                    y=data,
                    name=name,
                    box_visible=True,
                    line_color=color,
                    meanline_visible=True
                ))
                
            fig.update_layout(
                title=f"{selected_feature} Distribution by Preference Level",
                yaxis_title=selected_feature
            )
            st.plotly_chart(fig)
            
            # Correlation matrix for audio features
            st.subheader("Feature Correlations")
            corr_matrix = training_df[feature_cols].corr()
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig)
            
        with tab3:
            st.subheader("Musical Journey Timeline")
            
            # Convert dates and handle invalid formats
            training_df['Release Date'] = pd.to_datetime(training_df['Release Date'], errors='coerce')
            
            # Drop rows with invalid dates
            training_df = training_df.dropna(subset=['Release Date'])
            
            # Create timeline
            timeline_data = training_df.groupby([
                pd.Grouper(key='Release Date', freq='Y'),
                'playlist_origin'
            ]).size().reset_index(name='count')
            
            fig = px.line(
                timeline_data,
                x='Release Date',
                y='count',
                color='playlist_origin',
                title="Music Preferences Over Time",
                labels={
                    'playlist_origin': 'Preference Level',
                    'count': 'Number of Songs',
                    'Release Date': 'Year'
                }
            )
            st.plotly_chart(fig)
            
            # Artist network centrality analysis
            st.subheader("Top Artists by Centrality")
            top_artists = training_df[training_df['playlist_origin'] == 'df_liked'].nlargest(10, 'Artist Centrality')
            
            fig = px.bar(
                top_artists,
                x='Artist Name(s)',
                y='Artist Centrality',
                title="Most Central Artists in Mike's Music Network",
                labels={
                    'Artist Name(s)': 'Artist',
                    'Artist Centrality': 'Network Centrality Score'
                }
            )
            st.plotly_chart(fig)
    
    else:  # Music Analytics
        st.title("📊 Music Analytics Dashboard")
        
        # Distribution of predictions
        st.subheader("Prediction Distribution")
        fig = px.histogram(
            df,
            x='avg_score',
            nbins=30,
            title="Distribution of Prediction Scores"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Genre analysis
        st.subheader("Genre Performance Analysis")
        
        # Explode genres and calculate average scores
        genre_data = df.assign(
            Genres=df['Genres'].str.split(',')
        ).explode('Genres')
        genre_data['Genres'] = genre_data['Genres'].str.strip()
        
        genre_scores = genre_data.groupby('Genres').agg({
            'avg_score': 'mean',
            'Album Name': 'count'
        }).reset_index()
        
        genre_scores = genre_scores[genre_scores['Album Name'] > 1].sort_values('avg_score', ascending=False)
        
        fig = px.bar(
            genre_scores,
            x='Genres',
            y='avg_score',
            title="Average Prediction Score by Genre",
            color='Album Name',
            labels={'avg_score': 'Average Score', 'Album Name': 'Number of Albums'}
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()