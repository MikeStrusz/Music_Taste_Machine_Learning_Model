import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Setup Spotify API
client_id = 'your_client_id'
client_secret = 'your_client_secret'
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# Streamlit app
st.title("CSV Music Analyzer & Recommender")

# CSV file upload
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(uploaded_file)

    st.write("Here's the data you've uploaded:")
    st.dataframe(df.head())

    # Set 'genre' column to 'NA' if it's missing or has all null values
    if 'genre' not in df.columns or df['genre'].isnull().all():
        genre = 'NA'
        st.write("No 'genre' column found, or it contains only missing values. Using 'NA' for recommendations.")
    else:
        genre = df['genre'].mode()[0] if not df['genre'].isnull().all() else 'NA'
        st.write(f"Most common genres: {df['genre'].value_counts().head()}")

    if genre != 'NA':
        st.write(f"Recommending music similar to {genre}...")

        # Search for related music on Spotify
        results = sp.search(q=f'genre:{genre}', type='track', limit=5)
        tracks = results['tracks']['items']

        for track in tracks:
            st.write(f"Song: {track['name']}, Artist: {track['artists'][0]['name']}")
            st.write(f"Link: [Play on Spotify](https://open.spotify.com/track/{track['id']})")
    else:
        st.write("Could not determine a valid genre for recommendations.")
