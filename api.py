# api.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import glob
from datetime import datetime
from typing import List, Optional
import math
import random
from pydantic import BaseModel
import networkx as nx
from functools import lru_cache

app = FastAPI(title="Mike's Album Rankings API")

# Allow requests from your React frontend (will be http://localhost:5173 during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],   # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve covers from the local 'covers' folder at /covers/...
if os.path.exists("covers"):
    app.mount("/covers", StaticFiles(directory="covers"), name="covers")

# -------------------------------------------------------------------
# Helper functions (adapted from your app.py)
# -------------------------------------------------------------------

import networkx as nx
from functools import lru_cache
import glob

def build_graph(include_nmf=False):
    """
    Build a graph of artists based on similarity data.
    Nodes: all artists from training data, all artists from predictions, and all artists from similar artists file.
    Edges: between an artist and their similar artists (if available).
    """
    G = nx.Graph()

    # 1. Load training data to get all artists that appear there
    training_file = 'data/2026_training_complete_with_features.csv'
    all_artists = set()
    if os.path.exists(training_file):
        df = pd.read_csv(training_file)
        # Artist Name(s) column may contain multiple artists separated by commas or semicolons
        artists_series = df['Artist Name(s)'].dropna().str.split(r'[;,]+').explode().str.strip()
        all_artists.update(artists_series)

    # 2. Add artists from all prediction files
    prediction_files = glob.glob('predictions/*_Album_Recommendations.csv')
    for file in prediction_files:
        try:
            df_pred = pd.read_csv(file)
            if 'Artist Name(s)' in df_pred.columns:
                artists = df_pred['Artist Name(s)'].dropna().str.split(r'[;,]+').explode().str.strip()
                all_artists.update(artists)
        except:
            continue

    # Add all artists as nodes (with a default type)
    for artist in all_artists:
        G.add_node(artist, type='general')

    # 3. Add edges from similar artists file
    similar_df = pd.DataFrame()
    similar_file = 'data/liked_artists_only_similar.csv'
    if os.path.exists(similar_file):
        similar_df = pd.read_csv(similar_file)
        for _, row in similar_df.iterrows():
            artist = row.get('Artist', '')
            if not artist:
                continue
            similar_str = row.get('Similar Artists', '')
            if pd.isna(similar_str):
                continue
            similar_list = [s.strip() for s in similar_str.split(',')]
            # Ensure the main artist is in the graph
            if artist not in G:
                G.add_node(artist, type='similar_source')
            for s in similar_list:
                if s not in G:
                    G.add_node(s, type='similar_target')
                G.add_edge(artist, s)

    # Ensure Lucy Dacus is in the graph
    if "Lucy Dacus" not in G:
        G.add_node("Lucy Dacus", type='central')

    if include_nmf:
        # Optionally add nodes for NMF and not-liked artists (if you want them, but they are already included above)
        pass

    return G

@lru_cache(maxsize=1)
def get_graph():
    """Cached graph (build once)"""
    return build_graph(include_nmf=True)

@app.get("/graph/artists")
def list_artists():
    """Return list of all artist names in the graph (for autocomplete)."""
    G = get_graph()
    return sorted(list(G.nodes()))

def get_all_prediction_files():
    """Return list of (file_path, date_obj, formatted_date) for weeks >= Feb 2025"""
    prediction_files = glob.glob('predictions/*_Album_Recommendations.csv')
    cutoff = datetime(2025, 2, 1)
    results = []
    for f in prediction_files:
        date_str = os.path.basename(f).split('_')[0]
        try:
            date_obj = datetime.strptime(date_str, '%m-%d-%y')
            if date_obj >= cutoff:
                formatted = date_obj.strftime('%B %d, %Y')
                results.append((f, date_obj, formatted))
        except:
            continue
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def load_predictions(file_path: str):
    """Load predictions from a specific file, return dict and analysis date"""
    df = pd.read_csv(file_path)
    # Standardise column names
    if 'Album' in df.columns:
        df['Album Name'] = df['Album']
    elif 'Album Name' in df.columns:
        df['Album'] = df['Album Name']
    if 'Artist Name(s)' in df.columns:
        df['Artist'] = df['Artist Name(s)']
    if 'Predicted_Score' in df.columns:
        df['avg_score'] = df['Predicted_Score']
    if 'playlist_origin' not in df.columns:
        df['playlist_origin'] = 'unknown'
    df = df.drop_duplicates(subset=['Artist', 'Album'])
    date_str = os.path.basename(file_path).split('_')[0]
    analysis_date = datetime.strptime(date_str, '%m-%d-%y').strftime('%Y-%m-%d')
    # Return as list of records for JSON
    return df.to_dict(orient='records'), analysis_date

def load_master_gut_scores():
    """Load master gut scores file"""
    master_file = 'feedback/master_gut_scores.csv'
    if os.path.exists(master_file):
        df = pd.read_csv(master_file)
        return df.to_dict(orient='records')
    return []

# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/weeks")
def list_weeks():
    """Return list of available weeks with their file paths"""
    files = get_all_prediction_files()
    return [
        {"formatted": formatted, "file": file}
        for file, _, formatted in files
    ]

import math

def clean_nan(obj):
    """Recursively replace NaN with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj

@app.get("/albums/{week}")
def get_albums(week: str):
    """
    Get albums for a given week.
    week can be the formatted date (e.g., "April 12, 2025") or the filename.
    """
    try:
        files = get_all_prediction_files()
        target_file = None
        for file, _, formatted in files:
            if week == formatted or week == file:
                target_file = file
                break
        if not target_file:
            raise HTTPException(status_code=404, detail="Week not found")
        records, analysis_date = load_predictions(target_file)

        # Filter out nuked albums
        nuke_file = 'data/nuked_albums.csv'
        if os.path.exists(nuke_file):
            nuked_df = pd.read_csv(nuke_file)
            nuked_set = set(zip(nuked_df['Artist'], nuked_df['Album Name']))
            records = [r for r in records if (r.get('Artist'), r.get('Album')) not in nuked_set]

        # Load album covers
        covers_df = pd.DataFrame()
        covers_file = 'data/nmf_album_covers.csv'
        if os.path.exists(covers_file):
            covers_df = pd.read_csv(covers_file)
            # Normalize columns
            if 'Artist Name(s)' in covers_df.columns:
                covers_df = covers_df.rename(columns={'Artist Name(s)': 'Artist'})
            if 'Album Name' in covers_df.columns and 'Album' not in covers_df.columns:
                covers_df = covers_df.rename(columns={'Album Name': 'Album'})
            covers_df = covers_df.drop_duplicates(subset=['Artist', 'Album'], keep='first')
            # Convert cover paths to URLs – handle NaN and both slash types
            def fix_cover_path(val):
                if pd.isna(val):
                    return None
                if not isinstance(val, str):
                    return None
                # Normalize backslashes to forward slashes
                val = val.replace('\\', '/')
                if val.startswith('covers/'):
                    return f"/{val}"
                return val
            covers_df['Album Art'] = covers_df['Album Art'].apply(fix_cover_path)

        # Load album links (Spotify URLs)
        links_df = pd.DataFrame()
        links_file1 = 'data/nmf_album_links.csv'
        links_file2 = 'data/album_metadata_cache.csv'
        if os.path.exists(links_file1):
            links_df1 = pd.read_csv(links_file1)
            if 'Artist Name(s)' in links_df1.columns:
                links_df1 = links_df1.rename(columns={'Artist Name(s)': 'Artist'})
            if 'Album Name' in links_df1.columns:
                links_df1 = links_df1.rename(columns={'Album Name': 'Album'})
            links_df = links_df1[['Artist', 'Album', 'Spotify URL']].copy()
        if os.path.exists(links_file2):
            links_df2 = pd.read_csv(links_file2)
            # Assume columns: artist, album, spotify_url
            links_df2 = links_df2.rename(columns={'artist': 'Artist', 'album': 'Album', 'spotify_url': 'Spotify URL'})
            if 'Artist' in links_df2.columns and 'Album' in links_df2.columns:
                links_df2 = links_df2[['Artist', 'Album', 'Spotify URL']]
                links_df = pd.concat([links_df, links_df2], ignore_index=True)
        links_df = links_df.drop_duplicates(subset=['Artist', 'Album'], keep='last')

        # Merge covers and links into records
        for r in records:
            # Merge covers
            cover_match = covers_df[(covers_df['Artist'] == r['Artist']) & (covers_df['Album'] == r['Album'])]
            if not cover_match.empty:
                r['Album Art'] = cover_match.iloc[0]['Album Art']
            else:
                r['Album Art'] = None
            # Merge links
            link_match = links_df[(links_df['Artist'] == r['Artist']) & (links_df['Album'] == r['Album'])]
            if not link_match.empty:
                r['Spotify URL'] = link_match.iloc[0]['Spotify URL']
            else:
                r['Spotify URL'] = None

        # Load similar artists
        similar_df = pd.DataFrame()
        similar_file = 'data/liked_artists_only_similar.csv'
        if os.path.exists(similar_file):
            similar_df = pd.read_csv(similar_file)
            # Normalize column names (expected: Artist, Similar Artists)
            if 'Artist' not in similar_df.columns:
                # fallback: maybe the file has different names, but assume it's correct
                pass
            similar_df = similar_df.drop_duplicates(subset=['Artist'], keep='first')

        for r in records:
            # Attach similar artists
            sim_match = similar_df[similar_df['Artist'] == r['Artist']]
            if not sim_match.empty:
                r['Similar Artists'] = sim_match.iloc[0]['Similar Artists']
            else:
                r['Similar Artists'] = None

        # Attach gut scores if any
        gut_scores = load_master_gut_scores()
        gut_dict = { (g['Artist'], g['Album']): g for g in gut_scores if 'Artist' in g and 'Album' in g }
        for r in records:
            key = (r.get('Artist'), r.get('Album'))
            if key in gut_dict:
                r['gut_score'] = gut_dict[key]['gut_score']
                r['notes'] = gut_dict[key].get('notes', '')

        # Stamp prediction year onto each record
        date_str_raw = os.path.basename(target_file).split('_')[0]
        pred_year = '20' + date_str_raw.split('-')[2]
        for r in records:
            r['prediction_year'] = pred_year

        # Clean NaN values for JSON
        records = clean_nan(records)

        return {"week": analysis_date, "albums": records}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/top100")
def top100():
    """Return current Top 100 albums from master gut scores (2026 only) with covers"""
    master = load_master_gut_scores()
    if not master:
        return []
    # Filter valid gut scores and current year
    df = pd.DataFrame(master)
    df = df[df['gut_score'].notna()]
    df['gut_score'] = pd.to_numeric(df['gut_score'], errors='coerce')
    df = df.dropna(subset=['gut_score'])
    if 'gut_score_date' in df.columns:
        df['gut_score_date'] = pd.to_datetime(df['gut_score_date'], errors='coerce')
        current_year = datetime.now().year
        df = df[df['gut_score_date'].dt.year == current_year]
    df = df.sort_values('gut_score', ascending=False).head(100)

    # Load covers to attach
    covers_df = pd.DataFrame()
    covers_file = 'data/nmf_album_covers.csv'
    if os.path.exists(covers_file):
        covers_df = pd.read_csv(covers_file)
        if 'Artist Name(s)' in covers_df.columns:
            covers_df = covers_df.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in covers_df.columns and 'Album' not in covers_df.columns:
            covers_df = covers_df.rename(columns={'Album Name': 'Album'})
        covers_df = covers_df.drop_duplicates(subset=['Artist', 'Album'], keep='first')
        # Convert cover paths to URLs (same as in get_albums)
        def fix_cover_path(val):
            if pd.isna(val):
                return None
            if not isinstance(val, str):
                return None
            val = val.replace('\\', '/')
            if val.startswith('covers/'):
                return f"/{val}"
            return val
        covers_df['Album Art'] = covers_df['Album Art'].apply(fix_cover_path)

    # Merge covers into df
    if not covers_df.empty:
        df = df.merge(covers_df[['Artist', 'Album', 'Album Art']], on=['Artist', 'Album'], how='left')

    # Load album links (Spotify URLs)
    links_df = pd.DataFrame()
    links_file1 = 'data/nmf_album_links.csv'
    links_file2 = 'data/album_metadata_cache.csv'
    if os.path.exists(links_file1):
        links_df1 = pd.read_csv(links_file1)
        if 'Artist Name(s)' in links_df1.columns:
            links_df1 = links_df1.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in links_df1.columns:
            links_df1 = links_df1.rename(columns={'Album Name': 'Album'})
        links_df = links_df1[['Artist', 'Album', 'Spotify URL']].copy()
    if os.path.exists(links_file2):
        links_df2 = pd.read_csv(links_file2)
        links_df2 = links_df2.rename(columns={'artist': 'Artist', 'album': 'Album', 'spotify_url': 'Spotify URL'})
        if 'Artist' in links_df2.columns and 'Album' in links_df2.columns:
            links_df2 = links_df2[['Artist', 'Album', 'Spotify URL']]
            links_df = pd.concat([links_df, links_df2], ignore_index=True)
    links_df = links_df.drop_duplicates(subset=['Artist', 'Album'], keep='last')

    # Merge links into df
    if not links_df.empty:
        df = df.merge(links_df[['Artist', 'Album', 'Spotify URL']], on=['Artist', 'Album'], how='left')

    # Stamp prediction year from gut_score_date
    if 'gut_score_date' in df.columns:
        df['prediction_year'] = pd.to_datetime(df['gut_score_date'], errors='coerce').dt.year.astype('Int64').astype(str).replace('<NA>', None)

    records = df.to_dict(orient='records')
    return clean_nan(records)

@app.get("/random/old_fav")
def random_old_fav():
    """Return a random 'Old Fav' album from top 100 or previous years' gut scores"""
    # Load training data
    training_file = 'data/2026_training_complete_with_features.csv'
    if not os.path.exists(training_file):
        raise HTTPException(status_code=500, detail="Training file not found")
    training_df = pd.read_csv(training_file)

    # Top 50 from training
    top50 = training_df[
        (training_df['source_type'] == 'top_100_ranked') &
        (training_df['liked'] >= 80)
    ][['Album Name', 'Artist Name(s)']].drop_duplicates()
    candidates = []
    for _, row in top50.iterrows():
        artist = row['Artist Name(s)'].split(';')[0].split(',')[0].strip()
        album = row['Album Name']
        candidates.append({'Artist': artist, 'Album': album})

    # Load gut scores from previous years
    master_file = 'feedback/master_gut_scores.csv'
    if os.path.exists(master_file):
        gut_df = pd.read_csv(master_file)
        gut_df['gut_score_date'] = pd.to_datetime(gut_df['gut_score_date'], errors='coerce')
        current_year = datetime.now().year
        high_gut = gut_df[
            (gut_df['gut_score'] >= 80) &
            (gut_df['gut_score_date'].dt.year < current_year)
        ]
        for _, row in high_gut.iterrows():
            candidates.append({'Artist': row['Artist'], 'Album': row['Album']})

    if not candidates:
        raise HTTPException(status_code=404, detail="No Old Fav albums found")

    # Pick random candidate
    pick = random.choice(candidates)
    artist = pick['Artist']
    album = pick['Album']

    # Enrich with metadata (covers, links, similar artists, gut score)
    # Load covers
    covers_df = pd.DataFrame()
    covers_file = 'data/nmf_album_covers.csv'
    if os.path.exists(covers_file):
        covers_df = pd.read_csv(covers_file)
        if 'Artist Name(s)' in covers_df.columns:
            covers_df = covers_df.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in covers_df.columns and 'Album' not in covers_df.columns:
            covers_df = covers_df.rename(columns={'Album Name': 'Album'})
        covers_df = covers_df.drop_duplicates(subset=['Artist', 'Album'], keep='first')
        def fix_cover_path(val):
            if pd.isna(val):
                return None
            if not isinstance(val, str):
                return None
            val = val.replace('\\', '/')
            if val.startswith('covers/'):
                return f"/{val}"
            return val
        covers_df['Album Art'] = covers_df['Album Art'].apply(fix_cover_path)

    # Load links
    links_df = pd.DataFrame()
    links_file1 = 'data/nmf_album_links.csv'
    links_file2 = 'data/album_metadata_cache.csv'
    if os.path.exists(links_file1):
        links_df1 = pd.read_csv(links_file1)
        if 'Artist Name(s)' in links_df1.columns:
            links_df1 = links_df1.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in links_df1.columns:
            links_df1 = links_df1.rename(columns={'Album Name': 'Album'})
        links_df = links_df1[['Artist', 'Album', 'Spotify URL']].copy()
    if os.path.exists(links_file2):
        links_df2 = pd.read_csv(links_file2)
        links_df2 = links_df2.rename(columns={'artist': 'Artist', 'album': 'Album', 'spotify_url': 'Spotify URL'})
        if 'Artist' in links_df2.columns and 'Album' in links_df2.columns:
            links_df2 = links_df2[['Artist', 'Album', 'Spotify URL']]
            links_df = pd.concat([links_df, links_df2], ignore_index=True)
    links_df = links_df.drop_duplicates(subset=['Artist', 'Album'], keep='last')

    # Load similar artists
    similar_df = pd.DataFrame()
    similar_file = 'data/liked_artists_only_similar.csv'
    if os.path.exists(similar_file):
        similar_df = pd.read_csv(similar_file)
        similar_df = similar_df.drop_duplicates(subset=['Artist'], keep='first')

    # Build album object
    album_obj = {
        'Artist': artist,
        'Album': album,
        'Album Name': album,
        'avg_score': None,
        'gut_score': None,
        'notes': None,
        'Genres': None,
        'Label': None,
        'Release Date': None,
        'Similar Artists': None,
        'Spotify URL': None
    }

    # Add metadata from covers
    cover_match = covers_df[(covers_df['Artist'] == artist) & (covers_df['Album'] == album)]
    if not cover_match.empty:
        album_obj['Album Art'] = cover_match.iloc[0]['Album Art']
    else:
        album_obj['Album Art'] = None

    # Add links
    link_match = links_df[(links_df['Artist'] == artist) & (links_df['Album'] == album)]
    if not link_match.empty:
        album_obj['Spotify URL'] = link_match.iloc[0]['Spotify URL']

    # Add similar artists
    sim_match = similar_df[similar_df['Artist'] == artist]
    if not sim_match.empty:
        album_obj['Similar Artists'] = sim_match.iloc[0]['Similar Artists']

    # Add gut score from master
    gut_scores = load_master_gut_scores()
    gut_dict = { (g['Artist'], g['Album']): g for g in gut_scores if 'Artist' in g and 'Album' in g }
    key = (artist, album)
    if key in gut_dict:
        album_obj['gut_score'] = gut_dict[key]['gut_score']
        album_obj['notes'] = gut_dict[key].get('notes', '')

    # For old fav, also include the original liked score from training
    match = training_df[
        (training_df['Artist Name(s)'].str.contains(artist, na=False, case=False)) &
        (training_df['Album Name'] == album)
    ]
    if not match.empty:
        album_obj['avg_score'] = match.iloc[0]['liked']
        album_obj['Genres'] = match.iloc[0].get('Genres', None)
        album_obj['Label'] = match.iloc[0].get('Record Label', None)
        album_obj['Release Date'] = match.iloc[0].get('Release Date', None)

    return clean_nan(album_obj)

@app.get("/random/new_potential")
def random_new_potential():
    """Return a random 'Hidden Gem' (predicted >=75, not rated)"""
    # Load all prediction files
    prediction_files = get_all_prediction_files()
    # Load master gut scores to exclude already rated
    gut_scores = load_master_gut_scores()
    gut_dict = { (g['Artist'], g['Album']): g for g in gut_scores if 'Artist' in g and 'Album' in g }
    # Also load training data to exclude already liked
    training_file = 'data/2026_training_complete_with_features.csv'
    training_df = pd.DataFrame()
    if os.path.exists(training_file):
        training_df = pd.read_csv(training_file)

    candidates = []
    for file, date_obj, _ in prediction_files:
        try:
            df = pd.read_csv(file)
            # Standardize columns
            if 'Album' in df.columns and 'Album Name' not in df.columns:
                df['Album Name'] = df['Album']
            if 'Artist Name(s)' in df.columns and 'Artist' not in df.columns:
                df['Artist'] = df['Artist Name(s)']
            score_col = 'Predicted_Score' if 'Predicted_Score' in df.columns else 'avg_score'
            for _, row in df.iterrows():
                score = row.get(score_col, 0)
                if pd.isna(score) or score < 75:
                    continue
                artist = str(row.get('Artist', '')).strip()
                album = str(row.get('Album', '')).strip()
                if not artist or not album:
                    continue
                # Skip if already rated or in training
                if (artist, album) in gut_dict:
                    continue
                if not training_df.empty:
                    match = training_df[
                        (training_df['Artist Name(s)'].str.contains(artist, na=False)) &
                        (training_df['Album Name'] == album)
                    ]
                    if not match.empty:
                        continue
                # Also skip if from current year and already rated? We'll rely on gut_dict.
                candidates.append({
                    'Artist': artist,
                    'Album': album,
                    'Predicted_Score': score,
                    'Genres': row.get('Genres', None),
                    'Label': row.get('Label', None),
                    'Release Date': row.get('Release Date', None),
                    'date_obj': date_obj
                })
        except Exception as e:
            continue

    if not candidates:
        raise HTTPException(status_code=404, detail="No new potential albums found")

    # Bias toward current year
    current_year = datetime.now().year
    this_year = [c for c in candidates if c['date_obj'].year == current_year]
    pool = this_year if this_year else candidates
    pick = random.choice(pool)

    # Enrich with metadata (covers, links, similar)
    # Load covers, links, similar (same as before – we'll reuse the same code but to avoid duplication, we'll call a helper function)
    # For brevity, I'll repeat the enrichment logic inline. But better to factor out.
    # Since we're adding endpoints, we can create a helper function `enrich_album(artist, album)` and use it.
    # However, to keep changes minimal, I'll repeat the enrichment code here.

    # We'll write a helper function later. For now, we'll duplicate the enrichment code from old_fav but with a few tweaks.
    # Let's first load all the dataframes we need outside to avoid reloading each time. But for simplicity, we'll repeat.

    # Load covers
    covers_df = pd.DataFrame()
    covers_file = 'data/nmf_album_covers.csv'
    if os.path.exists(covers_file):
        covers_df = pd.read_csv(covers_file)
        if 'Artist Name(s)' in covers_df.columns:
            covers_df = covers_df.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in covers_df.columns and 'Album' not in covers_df.columns:
            covers_df = covers_df.rename(columns={'Album Name': 'Album'})
        covers_df = covers_df.drop_duplicates(subset=['Artist', 'Album'], keep='first')
        def fix_cover_path(val):
            if pd.isna(val):
                return None
            if not isinstance(val, str):
                return None
            val = val.replace('\\', '/')
            if val.startswith('covers/'):
                return f"/{val}"
            return val
        covers_df['Album Art'] = covers_df['Album Art'].apply(fix_cover_path)

    # Load links
    links_df = pd.DataFrame()
    links_file1 = 'data/nmf_album_links.csv'
    links_file2 = 'data/album_metadata_cache.csv'
    if os.path.exists(links_file1):
        links_df1 = pd.read_csv(links_file1)
        if 'Artist Name(s)' in links_df1.columns:
            links_df1 = links_df1.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in links_df1.columns:
            links_df1 = links_df1.rename(columns={'Album Name': 'Album'})
        links_df = links_df1[['Artist', 'Album', 'Spotify URL']].copy()
    if os.path.exists(links_file2):
        links_df2 = pd.read_csv(links_file2)
        links_df2 = links_df2.rename(columns={'artist': 'Artist', 'album': 'Album', 'spotify_url': 'Spotify URL'})
        if 'Artist' in links_df2.columns and 'Album' in links_df2.columns:
            links_df2 = links_df2[['Artist', 'Album', 'Spotify URL']]
            links_df = pd.concat([links_df, links_df2], ignore_index=True)
    links_df = links_df.drop_duplicates(subset=['Artist', 'Album'], keep='last')

    # Load similar artists
    similar_df = pd.DataFrame()
    similar_file = 'data/liked_artists_only_similar.csv'
    if os.path.exists(similar_file):
        similar_df = pd.read_csv(similar_file)
        similar_df = similar_df.drop_duplicates(subset=['Artist'], keep='first')

    # Build album object
    artist = pick['Artist']
    album = pick['Album']
    album_obj = {
        'Artist': artist,
        'Album': album,
        'Album Name': album,
        'avg_score': pick['Predicted_Score'],
        'gut_score': None,
        'notes': None,
        'Genres': pick.get('Genres'),
        'Label': pick.get('Label'),
        'Release Date': pick.get('Release Date'),
        'Similar Artists': None,
        'Spotify URL': None
    }

    # Add metadata from covers
    cover_match = covers_df[(covers_df['Artist'] == artist) & (covers_df['Album'] == album)]
    if not cover_match.empty:
        album_obj['Album Art'] = cover_match.iloc[0]['Album Art']
    else:
        album_obj['Album Art'] = None

    # Add links
    link_match = links_df[(links_df['Artist'] == artist) & (links_df['Album'] == album)]
    if not link_match.empty:
        album_obj['Spotify URL'] = link_match.iloc[0]['Spotify URL']

    # Add similar artists
    sim_match = similar_df[similar_df['Artist'] == artist]
    if not sim_match.empty:
        album_obj['Similar Artists'] = sim_match.iloc[0]['Similar Artists']

    # Add gut score if any (unlikely since we filtered)
    key = (artist, album)
    if key in gut_dict:
        album_obj['gut_score'] = gut_dict[key]['gut_score']
        album_obj['notes'] = gut_dict[key].get('notes', '')

    return clean_nan(album_obj)

@app.get("/random/take_chance")
def random_take_chance():
    """Return a random album with predicted score 50-74, not rated"""
    # Similar to new_potential but with different score range
    prediction_files = get_all_prediction_files()
    gut_scores = load_master_gut_scores()
    gut_dict = { (g['Artist'], g['Album']): g for g in gut_scores if 'Artist' in g and 'Album' in g }
    training_file = 'data/2026_training_complete_with_features.csv'
    training_df = pd.DataFrame()
    if os.path.exists(training_file):
        training_df = pd.read_csv(training_file)

    candidates = []
    for file, date_obj, _ in prediction_files:
        try:
            df = pd.read_csv(file)
            if 'Album' in df.columns and 'Album Name' not in df.columns:
                df['Album Name'] = df['Album']
            if 'Artist Name(s)' in df.columns and 'Artist' not in df.columns:
                df['Artist'] = df['Artist Name(s)']
            score_col = 'Predicted_Score' if 'Predicted_Score' in df.columns else 'avg_score'
            for _, row in df.iterrows():
                score = row.get(score_col, 0)
                if pd.isna(score) or not (50 <= score <= 74):
                    continue
                artist = str(row.get('Artist', '')).strip()
                album = str(row.get('Album', '')).strip()
                if not artist or not album:
                    continue
                if (artist, album) in gut_dict:
                    continue
                if not training_df.empty:
                    match = training_df[
                        (training_df['Artist Name(s)'].str.contains(artist, na=False)) &
                        (training_df['Album Name'] == album)
                    ]
                    if not match.empty:
                        continue
                candidates.append({
                    'Artist': artist,
                    'Album': album,
                    'Predicted_Score': score,
                    'Genres': row.get('Genres', None),
                    'Label': row.get('Label', None),
                    'Release Date': row.get('Release Date', None),
                    'date_obj': date_obj
                })
        except Exception:
            continue

    if not candidates:
        raise HTTPException(status_code=404, detail="No take chance albums found")

    # Bias toward current year
    current_year = datetime.now().year
    this_year = [c for c in candidates if c['date_obj'].year == current_year]
    pool = this_year if this_year else candidates
    pick = random.choice(pool)

    # Enrich similarly (same as new_potential but we'll just reuse the code – to avoid duplication, we could factor out but for now repeat)
    # We'll reuse the same covers/links/similar loading and build album_obj.
    # To keep code clean, I'll assume the covers/links/similar dataframes are already loaded – but they are not in scope.
    # Actually, we can load them again; it's fine for now.
    covers_df = pd.DataFrame()
    covers_file = 'data/nmf_album_covers.csv'
    if os.path.exists(covers_file):
        covers_df = pd.read_csv(covers_file)
        if 'Artist Name(s)' in covers_df.columns:
            covers_df = covers_df.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in covers_df.columns and 'Album' not in covers_df.columns:
            covers_df = covers_df.rename(columns={'Album Name': 'Album'})
        covers_df = covers_df.drop_duplicates(subset=['Artist', 'Album'], keep='first')
        def fix_cover_path(val):
            if pd.isna(val):
                return None
            if not isinstance(val, str):
                return None
            val = val.replace('\\', '/')
            if val.startswith('covers/'):
                return f"/{val}"
            return val
        covers_df['Album Art'] = covers_df['Album Art'].apply(fix_cover_path)

    links_df = pd.DataFrame()
    links_file1 = 'data/nmf_album_links.csv'
    links_file2 = 'data/album_metadata_cache.csv'
    if os.path.exists(links_file1):
        links_df1 = pd.read_csv(links_file1)
        if 'Artist Name(s)' in links_df1.columns:
            links_df1 = links_df1.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in links_df1.columns:
            links_df1 = links_df1.rename(columns={'Album Name': 'Album'})
        links_df = links_df1[['Artist', 'Album', 'Spotify URL']].copy()
    if os.path.exists(links_file2):
        links_df2 = pd.read_csv(links_file2)
        links_df2 = links_df2.rename(columns={'artist': 'Artist', 'album': 'Album', 'spotify_url': 'Spotify URL'})
        if 'Artist' in links_df2.columns and 'Album' in links_df2.columns:
            links_df2 = links_df2[['Artist', 'Album', 'Spotify URL']]
            links_df = pd.concat([links_df, links_df2], ignore_index=True)
    links_df = links_df.drop_duplicates(subset=['Artist', 'Album'], keep='last')

    similar_df = pd.DataFrame()
    similar_file = 'data/liked_artists_only_similar.csv'
    if os.path.exists(similar_file):
        similar_df = pd.read_csv(similar_file)
        similar_df = similar_df.drop_duplicates(subset=['Artist'], keep='first')

    artist = pick['Artist']
    album = pick['Album']
    album_obj = {
        'Artist': artist,
        'Album': album,
        'Album Name': album,
        'avg_score': pick['Predicted_Score'],
        'gut_score': None,
        'notes': None,
        'Genres': pick.get('Genres'),
        'Label': pick.get('Label'),
        'Release Date': pick.get('Release Date'),
        'Similar Artists': None,
        'Spotify URL': None
    }

    cover_match = covers_df[(covers_df['Artist'] == artist) & (covers_df['Album'] == album)]
    if not cover_match.empty:
        album_obj['Album Art'] = cover_match.iloc[0]['Album Art']
    else:
        album_obj['Album Art'] = None

    link_match = links_df[(links_df['Artist'] == artist) & (links_df['Album'] == album)]
    if not link_match.empty:
        album_obj['Spotify URL'] = link_match.iloc[0]['Spotify URL']

    sim_match = similar_df[similar_df['Artist'] == artist]
    if not sim_match.empty:
        album_obj['Similar Artists'] = sim_match.iloc[0]['Similar Artists']

    key = (artist, album)
    if key in gut_dict:
        album_obj['gut_score'] = gut_dict[key]['gut_score']
        album_obj['notes'] = gut_dict[key].get('notes', '')

    return clean_nan(album_obj)

@app.get("/random/any")
def random_any():
    """Return a completely random album from any prediction week"""
    prediction_files = get_all_prediction_files()
    all_albums = []
    for file, _, _ in prediction_files:
        try:
            df = pd.read_csv(file)
            if 'Album' in df.columns and 'Album Name' not in df.columns:
                df['Album Name'] = df['Album']
            if 'Artist Name(s)' in df.columns and 'Artist' not in df.columns:
                df['Artist'] = df['Artist Name(s)']
            for _, row in df[['Artist', 'Album']].drop_duplicates().iterrows():
                artist = row['Artist']
                album = row['Album']
                all_albums.append({'Artist': artist, 'Album': album})
        except Exception:
            continue
    if not all_albums:
        raise HTTPException(status_code=404, detail="No albums found")
    pick = random.choice(all_albums)
    artist = pick['Artist']
    album = pick['Album']

    # Now enrich with all metadata (covers, links, similar, gut scores, predicted score, etc.)
    # We need to find which week this album came from to get its predicted score and other fields.
    # Let's search prediction files for the album to get its details.
    # Also load covers, links, similar as before.
    # We'll reuse the same enrichment code but we need to find the row with the album.
    # For simplicity, we'll search all prediction files for the first occurrence of the album.
    found_row = None
    for file, _, _ in prediction_files:
        try:
            df = pd.read_csv(file)
            if 'Album' in df.columns and 'Album Name' not in df.columns:
                df['Album Name'] = df['Album']
            if 'Artist Name(s)' in df.columns and 'Artist' not in df.columns:
                df['Artist'] = df['Artist Name(s)']
            match = df[(df['Artist'] == artist) & (df['Album'] == album)]
            if not match.empty:
                found_row = match.iloc[0]
                break
        except Exception:
            continue

    # Load covers, links, similar
    covers_df = pd.DataFrame()
    covers_file = 'data/nmf_album_covers.csv'
    if os.path.exists(covers_file):
        covers_df = pd.read_csv(covers_file)
        if 'Artist Name(s)' in covers_df.columns:
            covers_df = covers_df.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in covers_df.columns and 'Album' not in covers_df.columns:
            covers_df = covers_df.rename(columns={'Album Name': 'Album'})
        covers_df = covers_df.drop_duplicates(subset=['Artist', 'Album'], keep='first')
        def fix_cover_path(val):
            if pd.isna(val):
                return None
            if not isinstance(val, str):
                return None
            val = val.replace('\\', '/')
            if val.startswith('covers/'):
                return f"/{val}"
            return val
        covers_df['Album Art'] = covers_df['Album Art'].apply(fix_cover_path)

    links_df = pd.DataFrame()
    links_file1 = 'data/nmf_album_links.csv'
    links_file2 = 'data/album_metadata_cache.csv'
    if os.path.exists(links_file1):
        links_df1 = pd.read_csv(links_file1)
        if 'Artist Name(s)' in links_df1.columns:
            links_df1 = links_df1.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in links_df1.columns:
            links_df1 = links_df1.rename(columns={'Album Name': 'Album'})
        links_df = links_df1[['Artist', 'Album', 'Spotify URL']].copy()
    if os.path.exists(links_file2):
        links_df2 = pd.read_csv(links_file2)
        links_df2 = links_df2.rename(columns={'artist': 'Artist', 'album': 'Album', 'spotify_url': 'Spotify URL'})
        if 'Artist' in links_df2.columns and 'Album' in links_df2.columns:
            links_df2 = links_df2[['Artist', 'Album', 'Spotify URL']]
            links_df = pd.concat([links_df, links_df2], ignore_index=True)
    links_df = links_df.drop_duplicates(subset=['Artist', 'Album'], keep='last')

    similar_df = pd.DataFrame()
    similar_file = 'data/liked_artists_only_similar.csv'
    if os.path.exists(similar_file):
        similar_df = pd.read_csv(similar_file)
        similar_df = similar_df.drop_duplicates(subset=['Artist'], keep='first')

    album_obj = {
        'Artist': artist,
        'Album': album,
        'Album Name': album,
        'avg_score': None,
        'gut_score': None,
        'notes': None,
        'Genres': None,
        'Label': None,
        'Release Date': None,
        'Similar Artists': None,
        'Spotify URL': None
    }

    if found_row is not None:
        score_col = 'Predicted_Score' if 'Predicted_Score' in found_row else 'avg_score'
        album_obj['avg_score'] = found_row.get(score_col)
        album_obj['Genres'] = found_row.get('Genres')
        album_obj['Label'] = found_row.get('Label')
        album_obj['Release Date'] = found_row.get('Release Date')

    cover_match = covers_df[(covers_df['Artist'] == artist) & (covers_df['Album'] == album)]
    if not cover_match.empty:
        album_obj['Album Art'] = cover_match.iloc[0]['Album Art']
    else:
        album_obj['Album Art'] = None

    link_match = links_df[(links_df['Artist'] == artist) & (links_df['Album'] == album)]
    if not link_match.empty:
        album_obj['Spotify URL'] = link_match.iloc[0]['Spotify URL']

    sim_match = similar_df[similar_df['Artist'] == artist]
    if not sim_match.empty:
        album_obj['Similar Artists'] = sim_match.iloc[0]['Similar Artists']

    gut_scores = load_master_gut_scores()
    gut_dict = { (g['Artist'], g['Album']): g for g in gut_scores if 'Artist' in g and 'Album' in g }
    key = (artist, album)
    if key in gut_dict:
        album_obj['gut_score'] = gut_dict[key]['gut_score']
        album_obj['notes'] = gut_dict[key].get('notes', '')

    return clean_nan(album_obj)

import json

ORDER_FILE = 'data/top100_order.json'

@app.get("/top100/order")
def get_top100_order():
    if os.path.exists(ORDER_FILE):
        with open(ORDER_FILE, 'r') as f:
            return json.load(f)
    return []

@app.post("/top100/order")
def save_top100_order(order: List[str]):  # order is list of "Artist|Album" strings
    with open(ORDER_FILE, 'w') as f:
        json.dump(order, f)
    return {"status": "ok"}

@app.get("/graph/degrees/{artist}")
def degrees_to_lucy(artist: str):
    """Return the number of degrees between the given artist and Lucy Dacus."""
    G = get_graph()
    if artist not in G:
        raise HTTPException(status_code=404, detail="Artist not found in graph")
    try:
        path = nx.shortest_path(G, source=artist, target="Lucy Dacus")
        distance = len(path) - 1
        return {"artist": artist, "distance": distance, "path": path}
    except nx.NetworkXNoPath:
        return {"artist": artist, "distance": None, "path": None}

@app.get("/graph/path")
def path_between(from_artist: str, to_artist: str):
    """Return the shortest path between two artists."""
    G = get_graph()
    if from_artist not in G or to_artist not in G:
        raise HTTPException(status_code=404, detail="One or both artists not found")
    try:
        path = nx.shortest_path(G, source=from_artist, target=to_artist)
        distance = len(path) - 1
        return {"from": from_artist, "to": to_artist, "distance": distance, "path": path}
    except nx.NetworkXNoPath:
        return {"from": from_artist, "to": to_artist, "distance": None, "path": None}

from datetime import datetime
import pandas as pd
import os

class RateRequest(BaseModel):
    artist: str
    album: str
    score: int
    notes: str = ""

@app.post("/rate")
def rate_album(request: RateRequest):
    """Save gut score to master and weekly files"""
    album = request.album
    artist = request.artist
    score = request.score
    notes = request.notes
    if not (0 <= score <= 100):
        return {"status": "error", "message": "Score must be between 0 and 100"}

    # 1. Determine which weekly file this album belongs to
    prediction_files = get_all_prediction_files()
    source_file = None
    for file, _, _ in prediction_files:
        try:
            df = pd.read_csv(file)
            # Normalize column names
            if 'Album' in df.columns and 'Album Name' not in df.columns:
                df['Album Name'] = df['Album']
            if 'Artist Name(s)' in df.columns and 'Artist' not in df.columns:
                df['Artist'] = df['Artist Name(s)']
            # Check if album exists in this week's file
            if not df[(df['Artist'] == artist) & (df['Album Name'] == album)].empty:
                source_file = file
                break
        except Exception:
            continue

    if not source_file:
        # If not found in any prediction file, we can still save to master but warn
        print(f"Warning: Album {artist} - {album} not found in any prediction file. Saving only to master.")

    # 2. Save to weekly predictions file (if source_file exists)
    if source_file:
        try:
            predictions_df = pd.read_csv(source_file)
            # Normalize columns
            if 'Album' in predictions_df.columns and 'Album Name' not in predictions_df.columns:
                predictions_df['Album Name'] = predictions_df['Album']
            if 'Artist Name(s)' in predictions_df.columns and 'Artist' not in predictions_df.columns:
                predictions_df['Artist'] = predictions_df['Artist Name(s)']
            # Find the row and update
            mask = (predictions_df['Artist'] == artist) & (predictions_df['Album Name'] == album)
            if mask.any():
                predictions_df.loc[mask, 'gut_score'] = float(score)
                predictions_df.loc[mask, 'gut_score_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                predictions_df.loc[mask, 'notes'] = notes
                predictions_df.to_csv(source_file, index=False)
        except Exception as e:
            print(f"Error updating weekly file: {e}")

    # 3. Save to master gut scores file
    master_file = 'feedback/master_gut_scores.csv'
    os.makedirs('feedback', exist_ok=True)

    new_entry = {
        'Album': album,
        'Artist': artist,
        'gut_score': float(score),
        'gut_score_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_file': source_file or '',
        'notes': notes
    }

    if os.path.exists(master_file):
        master_df = pd.read_csv(master_file)
        # Remove any existing entry for this album
        mask = ~((master_df['Album'] == album) & (master_df['Artist'] == artist))
        master_df = master_df[mask]
        # Append new entry
        master_df = pd.concat([master_df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        master_df = pd.DataFrame([new_entry])

    master_df.to_csv(master_file, index=False)

    return {"status": "ok", "message": f"Saved {artist} - {album} with score {score}"}

class CoverRequest(BaseModel):
    artist: str
    album: str
    image_url: str

@app.post("/save_cover")
def save_cover(request: CoverRequest):
    """Save a cover image URL for an album"""
    covers_file = 'data/nmf_album_covers.csv'
    os.makedirs('data', exist_ok=True)
    
    # Load existing covers
    if os.path.exists(covers_file):
        covers_df = pd.read_csv(covers_file)
    else:
        covers_df = pd.DataFrame(columns=['Artist', 'Album Name', 'Album Art', 'status'])
    
    # Remove existing entry for this album
    covers_df = covers_df[~((covers_df['Artist'] == request.artist) & (covers_df['Album Name'] == request.album))]
    
    # Add new entry
    new_row = pd.DataFrame([{
        'Artist': request.artist,
        'Album Name': request.album,
        'Album Art': request.image_url,
        'status': None
    }])
    covers_df = pd.concat([covers_df, new_row], ignore_index=True)
    
    # Save back
    covers_df.to_csv(covers_file, index=False)
    
    return {"status": "ok", "message": f"Cover saved for {request.artist} - {request.album}"}

class NukeRequest(BaseModel):
    artist: str
    album: str

@app.post("/nuke")
def nuke_album(request: NukeRequest):
    """Add album to nuked_albums.csv"""
    nuke_file = 'data/nuked_albums.csv'
    os.makedirs('data', exist_ok=True)

    if os.path.exists(nuke_file):
        nuked_df = pd.read_csv(nuke_file)
    else:
        nuked_df = pd.DataFrame(columns=['Artist', 'Album Name', 'Reason'])

    if not nuked_df[(nuked_df['Artist'] == request.artist) & (nuked_df['Album Name'] == request.album)].empty:
        return {"status": "already_nuked", "message": "Album already nuked"}

    new_row = pd.DataFrame([{
        'Artist': request.artist,
        'Album Name': request.album,
        'Reason': 'Nuked via C shortcut'
    }])
    nuked_df = pd.concat([nuked_df, new_row], ignore_index=True)
    nuked_df.to_csv(nuke_file, index=False)

    return {"status": "ok", "message": f"Nuked {request.artist} - {request.album}"}

@app.get("/vault")
def get_vault():
    """Return all historical top 100 albums grouped by year"""
    vault_file = 'data/vault_clean.csv'
    if not os.path.exists(vault_file):
        return []
    df = pd.read_csv(vault_file)
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.sort_values(['Year', 'Rank'], ascending=[False, True])

    # Load covers
    covers_df = pd.DataFrame()
    covers_file = 'data/nmf_album_covers.csv'
    if os.path.exists(covers_file):
        covers_df = pd.read_csv(covers_file)
        if 'Artist Name(s)' in covers_df.columns:
            covers_df = covers_df.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in covers_df.columns and 'Album' not in covers_df.columns:
            covers_df = covers_df.rename(columns={'Album Name': 'Album'})
        covers_df = covers_df.drop_duplicates(subset=['Artist', 'Album'], keep='first')
        def fix_cover_path(val):
            if pd.isna(val): return None
            if not isinstance(val, str): return None
            val = val.replace('\\', '/')
            if val.startswith('covers/'): return f"/{val}"
            return val
        covers_df['Album Art'] = covers_df['Album Art'].apply(fix_cover_path)
        df = df.merge(covers_df[['Artist', 'Album', 'Album Art']], on=['Artist', 'Album'], how='left')

    records = df.to_dict(orient='records')
    return clean_nan(records)

@app.get("/discover/feed")
def discover_feed(limit: int = 20, offset: int = 0):
    """
    Returns a weighted mixed feed for the Discover page.
    ~60% unrated new potential (2026, predicted >= 65)
    ~20% take a chance (2026, predicted 40-64)
    ~20% old favs (high scored from training or previous gut scores)
    Excludes already-rated and nuked albums.
    """
    # Load vault for historical rank labels
    vault_df = pd.DataFrame()
    vault_file = 'data/vault_clean.csv'
    if os.path.exists(vault_file):
        vault_df = pd.read_csv(vault_file)
        vault_df['Artist'] = vault_df['Artist'].str.strip()
        vault_df['Album'] = vault_df['Album'].str.strip()

    # Load gut scores and nuked albums to exclude
    gut_scores = load_master_gut_scores()
    rated_set = set(
        (g['Artist'], g['Album']) for g in gut_scores
        if 'Artist' in g and 'Album' in g and g.get('gut_score') is not None
    )
    nuke_file = 'data/nuked_albums.csv'
    nuked_set = set()
    if os.path.exists(nuke_file):
        nuked_df = pd.read_csv(nuke_file)
        nuked_set = set(zip(nuked_df['Artist'], nuked_df['Album Name']))

    excluded = rated_set | nuked_set

    prediction_files = get_all_prediction_files()
    current_year = datetime.now().year

    new_potential = []
    take_a_chance = []

    for file, date_obj, _ in prediction_files:
        is_2026 = date_obj.year == current_year
        try:
            df = pd.read_csv(file)
            if 'Album' in df.columns and 'Album Name' not in df.columns:
                df['Album Name'] = df['Album']
            if 'Artist Name(s)' in df.columns and 'Artist' not in df.columns:
                df['Artist'] = df['Artist Name(s)']
            score_col = 'Predicted_Score' if 'Predicted_Score' in df.columns else 'avg_score'
            for _, row in df.iterrows():
                artist = str(row.get('Artist', '')).strip()
                album = str(row.get('Album', row.get('Album Name', ''))).strip()
                if not artist or not album:
                    continue
                if (artist, album) in excluded:
                    continue
                score = row.get(score_col, 0)
                if pd.isna(score):
                    score = 0
                entry = {
                    'Artist': artist,
                    'Album': album,
                    'avg_score': float(score),
                    'gut_score': None,
                    'notes': None,
                    'Genres': row.get('Genres'),
                    'Record Label': row.get('Record Label'),
                    'Release Date': row.get('Release Date'),
                    'Similar Artists': None,
                    'Spotify URL': None,
                    'Album Art': None,
                    'feed_type': None,
                    'is_2026': is_2026,
                    'prediction_year': str(date_obj.year),
                }
                if score >= 65:
                    new_potential.append(entry)
                elif score >= 40:
                    take_a_chance.append(entry)
        except Exception:
            continue

    # Old favs — from training data high scorers
    old_favs = []
    training_file = 'data/2026_training_complete_with_features.csv'
    if os.path.exists(training_file):
        training_df = pd.read_csv(training_file)
        top_training = training_df[
            (training_df['source_type'] == 'top_100_ranked') &
            (training_df['liked'] >= 75)
        ][['Album Name', 'Artist Name(s)', 'liked', 'Genres']].drop_duplicates()
        for _, row in top_training.iterrows():
            artist = str(row['Artist Name(s)']).split(';')[0].split(',')[0].strip()
            album = str(row['Album Name']).strip()
            if (artist, album) in excluded:
                continue
            vault_match = vault_df[
                (vault_df['Artist'] == artist) & (vault_df['Album'] == album)
            ] if not vault_df.empty else pd.DataFrame()
            vault_rank = int(vault_match.iloc[0]['Rank']) if not vault_match.empty else None
            vault_year = int(vault_match.iloc[0]['Year']) if not vault_match.empty else None
            vault_spotify = vault_match.iloc[0]['Spotify URL'] if not vault_match.empty else None

            old_favs.append({
                'Artist': artist,
                'Album': album,
                'avg_score': float(row['liked']),
                'gut_score': None,
                'notes': None,
                'Genres': row.get('Genres'),
                'Record Label': None,
                'Release Date': None,
                'Similar Artists': None,
                'Spotify URL': vault_spotify,
                'Album Art': None,
                'feed_type': 'old_fav',
                'is_2026': False,
                'vault_rank': vault_rank,
                'vault_year': vault_year,
            })

    # Deduplicate each pool
    def dedup(pool):
        seen = set()
        out = []
        for item in pool:
            k = (item['Artist'], item['Album'])
            if k not in seen:
                seen.add(k)
                out.append(item)
        return out

    new_potential = dedup(new_potential)
    take_a_chance = dedup(take_a_chance)
    old_favs = dedup(old_favs)

    # Bias 2026 to front but add randomness within year groups
    current_2026_new = [x for x in new_potential if x['is_2026']]
    older_new = [x for x in new_potential if not x['is_2026']]
    random.shuffle(current_2026_new)
    random.shuffle(older_new)
    new_potential = current_2026_new + older_new

    current_2026_chance = [x for x in take_a_chance if x['is_2026']]
    older_chance = [x for x in take_a_chance if not x['is_2026']]
    random.shuffle(current_2026_chance)
    random.shuffle(older_chance)
    take_a_chance = current_2026_chance + older_chance

    random.shuffle(old_favs)

    # Build weighted feed: 60% new potential, 20% take a chance, 20% old fav
    total = limit
    n_new = int(total * 0.6)
    n_chance = int(total * 0.2)
    n_old = total - n_new - n_chance

    # Apply offset for pagination
    pool_new = new_potential[offset * n_new: offset * n_new + n_new]
    pool_chance = take_a_chance[offset * n_chance: offset * n_chance + n_chance]
    pool_old = old_favs[offset * n_old: offset * n_old + n_old]

    # Tag feed types
    for item in pool_new:
        item['feed_type'] = 'new_potential'
    for item in pool_chance:
        item['feed_type'] = 'take_a_chance'
    for item in pool_old:
        item['feed_type'] = 'old_fav'

    # Interleave: new, new, new, chance, old, new, new, new, chance, old...
    feed = []
    i_new, i_chance, i_old = 0, 0, 0
    pattern = ['new', 'new', 'new', 'chance', 'old']
    while len(feed) < total:
        for slot in pattern:
            if len(feed) >= total:
                break
            if slot == 'new' and i_new < len(pool_new):
                feed.append(pool_new[i_new]); i_new += 1
            elif slot == 'chance' and i_chance < len(pool_chance):
                feed.append(pool_chance[i_chance]); i_chance += 1
            elif slot == 'old' and i_old < len(pool_old):
                feed.append(pool_old[i_old]); i_old += 1
        # If all pools exhausted, break
        if i_new >= len(pool_new) and i_chance >= len(pool_chance) and i_old >= len(pool_old):
            break

    # Enrich with covers, links, similar artists
    covers_df = pd.DataFrame()
    covers_file = 'data/nmf_album_covers.csv'
    if os.path.exists(covers_file):
        covers_df = pd.read_csv(covers_file)
        if 'Artist Name(s)' in covers_df.columns:
            covers_df = covers_df.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in covers_df.columns and 'Album' not in covers_df.columns:
            covers_df = covers_df.rename(columns={'Album Name': 'Album'})
        covers_df = covers_df.drop_duplicates(subset=['Artist', 'Album'], keep='first')
        def fix_cover_path(val):
            if pd.isna(val): return None
            if not isinstance(val, str): return None
            val = val.replace('\\', '/')
            if val.startswith('covers/'): return f"/{val}"
            return val
        covers_df['Album Art'] = covers_df['Album Art'].apply(fix_cover_path)

    links_df = pd.DataFrame()
    links_file1 = 'data/nmf_album_links.csv'
    if os.path.exists(links_file1):
        links_df1 = pd.read_csv(links_file1)
        if 'Artist Name(s)' in links_df1.columns:
            links_df1 = links_df1.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in links_df1.columns:
            links_df1 = links_df1.rename(columns={'Album Name': 'Album'})
        links_df = links_df1[['Artist', 'Album', 'Spotify URL']].copy()

    similar_df = pd.DataFrame()
    similar_file = 'data/liked_artists_only_similar.csv'
    if os.path.exists(similar_file):
        similar_df = pd.read_csv(similar_file)
        similar_df = similar_df.drop_duplicates(subset=['Artist'], keep='first')

    for item in feed:
        artist, album = item['Artist'], item['Album']
        cover_match = covers_df[(covers_df['Artist'] == artist) & (covers_df['Album'] == album)] if not covers_df.empty else pd.DataFrame()
        item['Album Art'] = cover_match.iloc[0]['Album Art'] if not cover_match.empty else None
        link_match = links_df[(links_df['Artist'] == artist) & (links_df['Album'] == album)] if not links_df.empty else pd.DataFrame()
        item['Spotify URL'] = link_match.iloc[0]['Spotify URL'] if not link_match.empty else None
        sim_match = similar_df[similar_df['Artist'] == artist] if not similar_df.empty else pd.DataFrame()
        item['Similar Artists'] = sim_match.iloc[0]['Similar Artists'] if not sim_match.empty else None

    return clean_nan(feed)

@app.get("/genres")
def list_genres():
    """Return all unique genres from prediction files and training data, sorted."""
    genres = set()

    prediction_files = get_all_prediction_files()
    for file, _, _ in prediction_files:
        try:
            df = pd.read_csv(file)
            if 'Genres' in df.columns:
                for val in df['Genres'].dropna():
                    for g in str(val).split(','):
                        g = g.strip()
                        if g:
                            genres.add(g)
        except Exception:
            continue

    training_file = 'data/2026_training_complete_with_features.csv'
    if os.path.exists(training_file):
        df = pd.read_csv(training_file)
        if 'Genres' in df.columns:
            for val in df['Genres'].dropna():
                for g in str(val).split(','):
                    g = g.strip()
                    if g:
                        genres.add(g)

    return sorted(genres, key=lambda x: x.lower())


@app.get("/discover/filter")
def discover_filter(genre: str, limit: int = 20, offset: int = 0):
    """
    Returns a weighted feed (same as /discover/feed) but only includes albums
    whose Genres field contains the given genre string (case‑insensitive substring).
    """
    # Load vault for historical rank labels
    vault_df = pd.DataFrame()
    vault_file = 'data/vault_clean.csv'
    if os.path.exists(vault_file):
        vault_df = pd.read_csv(vault_file)
        vault_df['Artist'] = vault_df['Artist'].str.strip()
        vault_df['Album'] = vault_df['Album'].str.strip()

    # Load gut scores and nuked albums to exclude
    gut_scores = load_master_gut_scores()
    rated_set = set(
        (g['Artist'], g['Album']) for g in gut_scores
        if 'Artist' in g and 'Album' in g and g.get('gut_score') is not None
    )
    nuke_file = 'data/nuked_albums.csv'
    nuked_set = set()
    if os.path.exists(nuke_file):
        nuked_df = pd.read_csv(nuke_file)
        nuked_set = set(zip(nuked_df['Artist'], nuked_df['Album Name']))

    excluded = rated_set | nuked_set

    prediction_files = get_all_prediction_files()
    current_year = datetime.now().year

    new_potential = []
    take_a_chance = []

    genre_lower = genre.lower().strip()

    for file, date_obj, _ in prediction_files:
        is_2026 = date_obj.year == current_year
        try:
            df = pd.read_csv(file)
            if 'Album' in df.columns and 'Album Name' not in df.columns:
                df['Album Name'] = df['Album']
            if 'Artist Name(s)' in df.columns and 'Artist' not in df.columns:
                df['Artist'] = df['Artist Name(s)']
            score_col = 'Predicted_Score' if 'Predicted_Score' in df.columns else 'avg_score'
            for _, row in df.iterrows():
                artist = str(row.get('Artist', '')).strip()
                album = str(row.get('Album', row.get('Album Name', ''))).strip()
                if not artist or not album:
                    continue
                if (artist, album) in excluded:
                    continue
                # Genre filter
                genres = row.get('Genres', '')
                if pd.isna(genres) or genre_lower not in str(genres).lower():
                    continue

                score = row.get(score_col, 0)
                if pd.isna(score):
                    score = 0
                entry = {
                    'Artist': artist,
                    'Album': album,
                    'avg_score': float(score),
                    'gut_score': None,
                    'notes': None,
                    'Genres': row.get('Genres'),
                    'Record Label': row.get('Record Label'),
                    'Release Date': row.get('Release Date'),
                    'Similar Artists': None,
                    'Spotify URL': None,
                    'Album Art': None,
                    'feed_type': None,
                    'is_2026': is_2026,
                    'prediction_year': str(date_obj.year),
                }
                if score >= 65:
                    new_potential.append(entry)
                elif score >= 40:
                    take_a_chance.append(entry)
        except Exception:
            continue

    # Old favs — from training data high scorers
    old_favs = []
    training_file = 'data/2026_training_complete_with_features.csv'
    if os.path.exists(training_file):
        training_df = pd.read_csv(training_file)
        # Filter training by genre
        genre_mask = training_df['Genres'].fillna('').str.lower().str.contains(genre_lower, na=False)
        top_training = training_df[
            (training_df['source_type'] == 'top_100_ranked') &
            (training_df['liked'] >= 75) &
            genre_mask
        ][['Album Name', 'Artist Name(s)', 'liked', 'Genres']].drop_duplicates()
        for _, row in top_training.iterrows():
            artist = str(row['Artist Name(s)']).split(';')[0].split(',')[0].strip()
            album = str(row['Album Name']).strip()
            if (artist, album) in excluded:
                continue
            vault_match = vault_df[
                (vault_df['Artist'] == artist) & (vault_df['Album'] == album)
            ] if not vault_df.empty else pd.DataFrame()
            vault_rank = int(vault_match.iloc[0]['Rank']) if not vault_match.empty else None
            vault_year = int(vault_match.iloc[0]['Year']) if not vault_match.empty else None
            vault_spotify = vault_match.iloc[0]['Spotify URL'] if not vault_match.empty else None

            old_favs.append({
                'Artist': artist,
                'Album': album,
                'avg_score': float(row['liked']),
                'gut_score': None,
                'notes': None,
                'Genres': row.get('Genres'),
                'Record Label': None,
                'Release Date': None,
                'Similar Artists': None,
                'Spotify URL': vault_spotify,
                'Album Art': None,
                'feed_type': 'old_fav',
                'is_2026': False,
                'vault_rank': vault_rank,
                'vault_year': vault_year,
            })

    # Deduplicate each pool
    def dedup(pool):
        seen = set()
        out = []
        for item in pool:
            k = (item['Artist'], item['Album'])
            if k not in seen:
                seen.add(k)
                out.append(item)
        return out

    new_potential = dedup(new_potential)
    take_a_chance = dedup(take_a_chance)
    old_favs = dedup(old_favs)

    # Bias 2026 to front but add randomness within year groups
    current_2026_new = [x for x in new_potential if x['is_2026']]
    older_new = [x for x in new_potential if not x['is_2026']]
    random.shuffle(current_2026_new)
    random.shuffle(older_new)
    new_potential = current_2026_new + older_new

    current_2026_chance = [x for x in take_a_chance if x['is_2026']]
    older_chance = [x for x in take_a_chance if not x['is_2026']]
    random.shuffle(current_2026_chance)
    random.shuffle(older_chance)
    take_a_chance = current_2026_chance + older_chance

    random.shuffle(old_favs)

    # Build weighted feed: 60% new potential, 20% take a chance, 20% old fav
    total = limit
    n_new = int(total * 0.6)
    n_chance = int(total * 0.2)
    n_old = total - n_new - n_chance

    # Apply offset for pagination
    pool_new = new_potential[offset * n_new: offset * n_new + n_new]
    pool_chance = take_a_chance[offset * n_chance: offset * n_chance + n_chance]
    pool_old = old_favs[offset * n_old: offset * n_old + n_old]

    # Tag feed types
    for item in pool_new:
        item['feed_type'] = 'new_potential'
    for item in pool_chance:
        item['feed_type'] = 'take_a_chance'
    for item in pool_old:
        item['feed_type'] = 'old_fav'

    # Interleave: new, new, new, chance, old, new, new, new, chance, old...
    feed = []
    i_new, i_chance, i_old = 0, 0, 0
    pattern = ['new', 'new', 'new', 'chance', 'old']
    while len(feed) < total:
        for slot in pattern:
            if len(feed) >= total:
                break
            if slot == 'new' and i_new < len(pool_new):
                feed.append(pool_new[i_new]); i_new += 1
            elif slot == 'chance' and i_chance < len(pool_chance):
                feed.append(pool_chance[i_chance]); i_chance += 1
            elif slot == 'old' and i_old < len(pool_old):
                feed.append(pool_old[i_old]); i_old += 1
        if i_new >= len(pool_new) and i_chance >= len(pool_chance) and i_old >= len(pool_old):
            break

    # Enrich with covers, links, similar artists (same as in discover_feed)
    covers_df = pd.DataFrame()
    covers_file = 'data/nmf_album_covers.csv'
    if os.path.exists(covers_file):
        covers_df = pd.read_csv(covers_file)
        if 'Artist Name(s)' in covers_df.columns:
            covers_df = covers_df.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in covers_df.columns and 'Album' not in covers_df.columns:
            covers_df = covers_df.rename(columns={'Album Name': 'Album'})
        covers_df = covers_df.drop_duplicates(subset=['Artist', 'Album'], keep='first')
        def fix_cover_path(val):
            if pd.isna(val): return None
            if not isinstance(val, str): return None
            val = val.replace('\\', '/')
            if val.startswith('covers/'): return f"/{val}"
            return val
        covers_df['Album Art'] = covers_df['Album Art'].apply(fix_cover_path)

    links_df = pd.DataFrame()
    links_file1 = 'data/nmf_album_links.csv'
    if os.path.exists(links_file1):
        links_df1 = pd.read_csv(links_file1)
        if 'Artist Name(s)' in links_df1.columns:
            links_df1 = links_df1.rename(columns={'Artist Name(s)': 'Artist'})
        if 'Album Name' in links_df1.columns:
            links_df1 = links_df1.rename(columns={'Album Name': 'Album'})
        links_df = links_df1[['Artist', 'Album', 'Spotify URL']].copy()

    similar_df = pd.DataFrame()
    similar_file = 'data/liked_artists_only_similar.csv'
    if os.path.exists(similar_file):
        similar_df = pd.read_csv(similar_file)
        similar_df = similar_df.drop_duplicates(subset=['Artist'], keep='first')

    for item in feed:
        artist, album = item['Artist'], item['Album']
        cover_match = covers_df[(covers_df['Artist'] == artist) & (covers_df['Album'] == album)] if not covers_df.empty else pd.DataFrame()
        item['Album Art'] = cover_match.iloc[0]['Album Art'] if not cover_match.empty else None
        link_match = links_df[(links_df['Artist'] == artist) & (links_df['Album'] == album)] if not links_df.empty else pd.DataFrame()
        item['Spotify URL'] = link_match.iloc[0]['Spotify URL'] if not link_match.empty else None
        sim_match = similar_df[similar_df['Artist'] == artist] if not similar_df.empty else pd.DataFrame()
        item['Similar Artists'] = sim_match.iloc[0]['Similar Artists'] if not sim_match.empty else None

    return clean_nan(feed)