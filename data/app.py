import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from typing import Tuple, List

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare all necessary datasets"""
    try:
        with st.spinner('Loading prediction data...'):
            # Get the most recent prediction file
            prediction_files = os.listdir('predictions')
            latest_prediction = sorted(
                [f for f in prediction_files if f.endswith('_Album_Recommendations.csv')],
                reverse=True
            )[0]
            
            predictions = pd.read_csv(f'predictions/{latest_prediction}')
            track_predictions = pd.read_csv('predictions/nmf_predictions_with_uncertainty.csv')
            feature_importance = pd.read_csv('data/feature_importance.csv')  # You'll need to save this during model training
            
            return predictions, track_predictions, feature_importance
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def create_network_visualization(G: nx.Graph) -> go.Figure:
    """Create an interactive network visualization using plotly"""
    pos = nx.spring_layout(G)
    
    edge_trace = go.Scatter(
        x=[], y=[], mode='lines',
        line=dict(width=0.5, color='#888'),
        hoverinfo='none'
    )
    
    node_trace = go.Scatter(
        x=[], y=[], mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
        )
    )
    
    # Add edges to visualization
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    # Add nodes to visualization
    node_trace['x'] = [pos[node][0] for node in G.nodes()]
    node_trace['y'] = [pos[node][1] for node in G.nodes()]
    node_trace['text'] = list(G.nodes())
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0,l=0,r=0,t=0),
                   ))
    return fig

def main():
    st.set_page_config(layout="wide", page_title="Music Recommendation Analysis")
    
    st.title("New Music Friday Prediction App")
    st.write("Predicting which new albums you'll enjoy based on your music taste!")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Choose a page", 
        ["Album Predictions", "Track Predictions", "Model Insights", "Artist Network", "About"]
    )
    
    # Load data with spinner
    with st.spinner('Loading data...'):
        album_predictions, track_predictions, feature_importance = load_data()
    
    if not all([album_predictions is not None, track_predictions is not None, feature_importance is not None]):
        st.error("Failed to load necessary data. Please check your data files.")
        return
    
    if page == "Album Predictions":
        st.header("Album Recommendations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Filter controls
            st.subheader("Filter Options")
            
            # Genre filter
            genres = album_predictions['Genres'].str.split(', ').explode().unique()
            selected_genres = st.multiselect("Filter by Genre", genres)
            
            # Confidence threshold
            confidence_threshold = st.slider(
                "Minimum Confidence Score", 
                min_value=0, 
                max_value=100, 
                value=50,
                help="Higher values indicate more reliable predictions"
            )
            
            # Search box
            search_term = st.text_input("Search by Artist or Album Name")
        
        with col2:
            # Sorting options
            sort_by = st.selectbox(
                "Sort by",
                ['weighted_score', 'avg_score', 'confidence_score', 'track_count']
            )
            sort_ascending = st.checkbox("Sort Ascending")
        
        # Apply filters
        filtered_predictions = album_predictions.copy()
        
        if selected_genres:
            filtered_predictions = filtered_predictions[
                filtered_predictions['Genres'].str.contains('|'.join(selected_genres), na=False)
            ]
        
        filtered_predictions = filtered_predictions[
            filtered_predictions['confidence_score'] >= confidence_threshold
        ]
        
        if search_term:
            search_mask = (
                filtered_predictions['Album Name'].str.contains(search_term, case=False, na=False) |
                filtered_predictions['Artist'].str.contains(search_term, case=False, na=False)
            )
            filtered_predictions = filtered_predictions[search_mask]
        
        # Sort results
        filtered_predictions = filtered_predictions.sort_values(
            sort_by, 
            ascending=sort_ascending
        )
        
        # Display results
        st.subheader("Filtered Results")
        st.dataframe(
            filtered_predictions[['Album Name', 'Artist', 'avg_score', 
                                'confidence_score', 'Genres', 'track_count']]
        )
        
        # Download button
        csv = filtered_predictions.to_csv(index=False)
        st.download_button(
            label="Download filtered results as CSV",
            data=csv,
            file_name="filtered_predictions.csv",
            mime="text/csv"
        )
        
        # Visualizations
        st.subheader("Prediction Visualization")
        
        viz_type = st.radio(
            "Choose visualization",
            ["Score vs Confidence", "Genre Distribution", "Score Distribution"]
        )
        
        if viz_type == "Score vs Confidence":
            fig = px.scatter(
                filtered_predictions,
                x='confidence_score',
                y='avg_score',
                hover_data=['Album Name', 'Artist', 'Genres'],
                title='Album Predictions: Score vs Confidence',
                color='track_count'
            )
            st.plotly_chart(fig)
            
        elif viz_type == "Genre Distribution":
            genre_counts = (
                filtered_predictions['Genres']
                .str.split(', ')
                .explode()
                .value_counts()
                .head(15)
            )
            fig = px.bar(
                x=genre_counts.index,
                y=genre_counts.values,
                title="Genre Distribution in Predictions"
            )
            st.plotly_chart(fig)
            
        else:  # Score Distribution
            fig = px.histogram(
                filtered_predictions,
                x='avg_score',
                title="Distribution of Prediction Scores"
            )
            st.plotly_chart(fig)
        
    elif page == "Track Predictions":
        st.header("Track-Level Predictions")
        
        # Add confidence interval visualization
        fig = go.Figure([
            go.Scatter(
                name='Upper Bound',
                x=track_predictions.index,
                y=track_predictions['prediction_upper'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='Prediction',
                x=track_predictions.index,
                y=track_predictions['predicted_score'],
                mode='lines',
                marker=dict(color="#ff0000"),
                line=dict(width=2),
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty'
            ),
            go.Scatter(
                name='Lower Bound',
                x=track_predictions.index,
                y=track_predictions['prediction_lower'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            )
        ])
        
        fig.update_layout(
            title='Track Predictions with Confidence Intervals',
            yaxis_title='Predicted Score',
            hovermode='x'
        )
        st.plotly_chart(fig)
        
    elif page == "Model Insights":
        st.header("Model Performance Insights")
        
        # Feature importance visualization
        st.subheader("Feature Importance")
        fig = px.bar(
            feature_importance.head(15),
            x='importance',
            y='feature',
            orientation='h',
            title="Top 15 Most Important Features"
        )
        st.plotly_chart(fig)
        
        # Add feature effect plots
        st.subheader("Feature Effects on Predictions")
        feature_to_analyze = st.selectbox(
            "Select feature to analyze",
            feature_importance['feature'].tolist()
        )
        
        # Create partial dependence plot
        # Note: You'll need to implement the actual partial dependence calculation
        
    elif page == "Artist Network":
        st.header("Artist Similarity Network")
        
        # Create and display network visualization
        # Note: You'll need to load your network data here
        st.write("Artist network visualization shows how artists in your music library are connected based on similarity.")
        
    else:  # About page
        st.header("About This Project")
        
        st.subheader("Methodology")
        st.write("""
        This project uses machine learning to predict your music preferences based on:
        1. Your historical listening data
        2. Audio features from Spotify
        3. Artist similarity data from Last.fm
        4. Genre information and record labels
        """)
        
        st.subheader("Features Used")
        st.write("""
        The model considers various features including:
        - Audio features (tempo, energy, danceability, etc.)
        - Artist network centrality
        - Genre similarity
        - Label performance
        """)
        
        st.subheader("Data Sources")
        st.write("""
        - Spotify API: Audio features and track metadata
        - Last.fm API: Artist similarity and genre information
        - Personal listening history
        - Record label information
        """)
        
        # Add model performance metrics
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="R-squared Score", 
                value="0.82",  # Replace with actual metrics
                help="Percentage of variance explained by the model"
            )
        
        with col2:
            st.metric(
                label="Mean Absolute Error", 
                value="12.3",  # Replace with actual metrics
                help="Average prediction error in score points"
            )

if __name__ == "__main__":
    main()