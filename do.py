import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from typing import Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyTrackAnalyzer:
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        """Initialize the Spotify client with authentication."""
        try:
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope='playlist-read-private playlist-read-collaborative user-library-read'
            ))
            logger.info("Successfully authenticated with Spotify")
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise

    def get_audio_features(self, track_id: str) -> Dict:
        """Retrieve audio features for a track with error handling."""
        try:
            features = self.sp.audio_features([track_id])[0]
            if features is None:
                logger.warning(f"No audio features found for track {track_id}")
                return {}
            
            return {
                'Danceability': features['danceability'],
                'Energy': features['energy'],
                'Key': features['key'],
                'Loudness': features['loudness'],
                'Mode': 'Major' if features['mode'] == 1 else 'Minor',
                'Speechiness': features['speechiness'],
                'Acousticness': features['acousticness'],
                'Instrumentalness': features['instrumentalness'],
                'Liveness': features['liveness'],
                'Valence': features['valence'],
                'Tempo': features['tempo'],
                'Time Signature': features['time_signature']
            }
        except Exception as e:
            logger.error(f"Error getting audio features for track {track_id}: {str(e)}")
            return {}

    def fetch_tracks(self, playlist_id: Optional[str] = None, liked_songs: bool = False) -> List[Dict]:
        """Fetch tracks from a playlist or liked songs with pagination support."""
        track_data = []
        try:
            if liked_songs:
                results = self.sp.current_user_saved_tracks()
            else:
                results = self.sp.playlist_tracks(playlist_id)
            
            while results:
                for item in results['items']:
                    track = item['track']
                    if track is None:
                        continue
                    
                    track_info = {
                        'Track ID': track['id'],
                        'Track Name': track['name'],
                        'Album Name': track['album']['name'],
                        'Artist Name(s)': ', '.join(artist['name'] for artist in track['artists']),
                        'Release Date': track['album']['release_date'],
                        'Duration (ms)': track['duration_ms'],
                        'Popularity': track['popularity'],
                        'Added By': item['added_by']['id'] if not liked_songs else 'N/A',
                        'Added At': item['added_at']
                    }
                    
                    # Get audio features and update track info
                    audio_features = self.get_audio_features(track['id'])
                    track_info.update(audio_features)
                    track_data.append(track_info)
                    
                    time.sleep(0.5)  # Rate limiting prevention
                
                # Get next set of results (pagination)
                results = self.sp.next(results) if results['next'] else None
                
            return track_data
            
        except Exception as e:
            logger.error(f"Error fetching tracks: {str(e)}")
            return []

def main():
    # Your Spotify API credentials
    client_id = 'e099aa20e771426eab189fa3840c2968'
    client_secret = '92092106f7c844ca98290aca5899aa2e'
    redirect_uri = 'http://localhost:8080/callback'
    
    try:
        # Initialize analyzer
        analyzer = SpotifyTrackAnalyzer(client_id, client_secret, redirect_uri)
        
        # Playlist IDs
        playlist_ids = {
            'nmf': '4sRLZd7rlGNaFYW1EbHHeI',
            'did_not_like': '3oeOC2CCCSsRWu6qkwS4Z4',
            '22to24': '7tEBkfh5pbsLMSnXYSueQ3'
        }
        
        # Fetch track data from all playlists
        all_track_data = []
        for playlist_name, playlist_id in playlist_ids.items():
            logger.info(f"Fetching tracks from playlist: {playlist_name}")
            playlist_tracks = analyzer.fetch_tracks(playlist_id=playlist_id)
            all_track_data.extend(playlist_tracks)
        
        # Fetch liked songs
        logger.info("Fetching liked songs")
        liked_tracks = analyzer.fetch_tracks(liked_songs=True)
        all_track_data.extend(liked_tracks)
        
        # Convert to DataFrame and save
        if all_track_data:
            df_tracks = pd.DataFrame(all_track_data)
            df_tracks.to_csv('spotify_tracks.csv', index=False)
            logger.info("Successfully saved track data to spotify_tracks.csv")
        else:
            logger.error("No track data was collected")
            
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")

if __name__ == "__main__":
    main()