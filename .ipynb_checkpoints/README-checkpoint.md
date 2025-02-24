# Music Taste Prediction Model: New Music Friday Recommender 💿🎧👍👎

This project is a machine learning model designed to predict my music taste based on my liked songs, recently loved albums, and albums I didn't enjoy. The model is trained using regression and recommends albums from Spotify's "New Music Friday" playlist. It incorporates features like artist centrality, genre encoding, mood scores, and energy profiles to make personalized recommendations.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Data Dictionary](#data-dictionary)
4. [Setup Instructions](#setup-instructions)
5. [Usage](#usage)
6. [Model Performance](#model-performance)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Overview

Hi, I'm Mike Strusz! 👋  
I'm a Data Analyst based in Milwaukee, passionate about solving real-world problems through data-driven insights. With a strong background in data analysis, visualization, and machine learning, I’m always expanding my skills to stay at the forefront of the field.

Before transitioning into data analytics, I spent over a decade as a teacher, where I developed a passion for making learning engaging and accessible. This experience has shaped my approach to data: breaking down complex concepts into understandable and actionable insights.

This project is, if I’m being honest, something I initially wanted for my own use. As an avid listener of contemporary music, I love evaluating and experiencing today’s best music, often attending concerts to immerse myself in the artistry. But beyond my personal interest, this project became a fascinating exploration of how machine learning can use past behavior to predict future preferences. It’s not about tracking listeners; it’s about understanding patterns and applying them to create better, more personalized experiences. This approach has broad applications, from music to e-commerce to customer segmentation, and it’s a powerful tool for any business looking to anticipate and meet customer needs.

The goal of this project is to predict my music preferences and recommend new albums from Spotify's "New Music Friday" playlist. The model uses a combination of regression techniques, network analysis, and feature engineering to make personalized recommendations. Key features include:

- **Artist Centrality**: Measures how central an artist is to my music taste using PageRank.
- **Genre Encoding**: Encodes genres based on their relevance to my preferences.
- **Mood Score**: Combines valence, danceability, and liveness to capture the vibe of a track.
- **Energy Profile**: Combines energy, loudness, and tempo to gauge the intensity of a track.

---

## Features

The model uses the following features to make predictions:

1. **Popularity**: The popularity score of the track (0-100).
2. **Genres_encoded**: Target-encoded genres based on my preferences.
3. **Artist Centrality**: A score representing how central an artist is to my music taste.
4. **Label_Category_encoded**: Target-encoded record label categories (Large, Medium, Small, Unknown/DIY).
5. **Mood Score**: A combination of valence, danceability, and liveness.
6. **Energy Profile**: A combination of energy, loudness, and tempo.
7. **Featured_Centrality_Score**: A score representing the centrality of featured artists.

---

## Data Dictionary

| Column Name                | Description                                                                 | Data Type  |
|----------------------------|-----------------------------------------------------------------------------|------------|
| Track ID                   | Unique identifier for the track.                                            | String     |
| Track Name                 | Name of the track.                                                          | String     |
| Album Name                 | Name of the album.                                                          | String     |
| Artist Name(s)             | Name(s) of the artist(s).                                                   | String     |
| Release Date               | Release date of the track.                                                  | Date       |
| Duration (ms)              | Duration of the track in milliseconds.                                      | Integer    |
| Popularity                 | Popularity score of the track (0-100).                                      | Integer    |
| Genres                     | Genres associated with the track.                                           | String     |
| Record Label               | Record label of the track.                                                  | String     |
| Danceability               | How suitable the track is for dancing (0-1).                                | Float      |
| Energy                     | Energy level of the track (0-1).                                            | Float      |
| Key                        | Key of the track.                                                           | Float      |
| Loudness                   | Loudness of the track in decibels (dB).                                     | Float      |
| Mode                       | Modality of the track (0 = Minor, 1 = Major).                               | Float      |
| Speechiness                | Presence of spoken words in the track (0-1).                                | Float      |
| Acousticness               | Acousticness of the track (0-1).                                            | Float      |
| Instrumentalness           | Instrumentalness of the track (0-1).                                        | Float      |
| Liveness                   | Presence of live audience sounds in the track (0-1).                        | Float      |
| Valence                    | Positivity of the track (0-1).                                              | Float      |
| Tempo                      | Tempo of the track in beats per minute (BPM).                               | Float      |
| liked                      | Target variable: 100 (liked), 50 (favorite albums), 0 (not liked).          | Float      |
| playlist_origin            | Source of the track (df_liked, df_fav_albums, df_not_liked, df_nmf).        | String     |
| Label_Category             | Size category of the record label (Large, Medium, Small, Unknown/DIY).      | String     |
| Label_Category_encoded     | Target-encoded record label category.                                       | Float      |
| is_unknown_genre           | Binary indicator for unknown genres.                                        | Integer    |
| Genres_encoded             | Target-encoded genres.                                                      | Float      |
| Featured_Artist(s)         | Name(s) of featured artists.                                                | String     |
| Artist Centrality          | PageRank score representing the artist's centrality in my music taste.      | Float      |
| Featured_Centrality_Score  | PageRank score for featured artists.                                        | Float      |
| mood_score                 | Combined score for valence, danceability, and liveness.                     | Float      |
| energy_profile             | Combined score for energy, loudness, and tempo.                             | Float      |

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/music-taste-prediction.git
   cd music-taste-prediction