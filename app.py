from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import logging

app = Flask(__name__)
CORS(app)  # This enables cross-origin requests, necessary for frontend integration.

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_recommendation_resources():
    """
    Load recommendation model and dataset with error handling
    """
    try:
        # Load the model
        model = joblib.load('nearest_neighbors_model.pkl')  
        
        # Load dataset
        dataset = pd.read_csv('interaction_data.csv')
        
        # Preprocess the data
        dataset['content'] = dataset['video_title'] + " " + \
                             dataset['video_description'] + " " + \
                             dataset['video_tags']
        
        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(dataset['content'])
        
        # Fit Nearest Neighbors model
        nn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
        nn_model.fit(tfidf_matrix)
        
        return {
            'model': model,
            'dataset': dataset,
            'tfidf_vectorizer': tfidf_vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'nn_model': nn_model
        }
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading recommendation resources: {e}")
        return None

# Load resources when the app starts
recommendation_resources = load_recommendation_resources()

@app.route('/recommend', methods=['GET'])
def recommend_videos():
    # Check if resources are loaded
    if recommendation_resources is None:
        return jsonify({"error": "Recommendation system is not properly initialized"}), 500
    
    # Unpack resources
    dataset = recommendation_resources['dataset']
    tfidf_matrix = recommendation_resources['tfidf_matrix']
    nn_model = recommendation_resources['nn_model']
    
    # Get parameters
    video_title = request.args.get('video_title', '').strip()
    num_recommendations = int(request.args.get('num_recommendations', 5))
    
    # Validate input
    if not video_title:
        return jsonify({"error": "Video title is required!"}), 400
    
    try:
        # Find video index
        matching_videos = dataset[dataset['video_title'].str.contains(video_title, case=False)]
        
        if matching_videos.empty:
            # If no exact match, try fuzzy matching
            return jsonify({
                "error": "No videos found matching the title.",
                "suggestions": dataset['video_title'].sample(min(5, len(dataset))).tolist()
            }), 404
        
        # Take the first matching video
        idx = matching_videos.index[0]
        
        # Get the nearest neighbors (similar videos)
        distances, indices = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=num_recommendations + 1)
        
        # Flatten indices and exclude the input video itself
        similar_indices = indices.flatten()[1:]
        
        # Get recommended videos
        recommended_videos = dataset.iloc[similar_indices][['video_title', 'video_link']].drop_duplicates().head(num_recommendations)
        
        if recommended_videos.empty:
            return jsonify({"error": "No sufficient recommendations available."}), 404
        
        recommendations = recommended_videos.to_dict(orient='records')
        return jsonify({"recommendations": recommendations})
    
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({"error": "An unexpected error occurred during recommendation."}), 500

@app.route('/search_titles', methods=['GET'])
def search_video_titles():
    """
    Endpoint to search for video titles (useful for autocomplete or suggestions)
    """
    if recommendation_resources is None:
        return jsonify({"error": "Recommendation system is not properly initialized"}), 500
    
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify({"error": "Search query is required"}), 400
    
    dataset = recommendation_resources['dataset']
    
    # Case-insensitive partial matching
    matching_titles = dataset[dataset['video_title'].str.contains(query, case=False)]['video_title'].unique()
    
    return jsonify({
        "titles": matching_titles.tolist()[:10]  # Limit to 10 suggestions
    })

if __name__ == '__main__':
    app.run(debug=True)