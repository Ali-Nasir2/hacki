from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Initialize Flask app
app = Flask(__name__)

# TMDb API details
API_KEY = '8c4019d99263475e2689b2e90d461b78'
BASE_URL = 'https://api.themoviedb.org/3'
IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/original'

def get_movie_poster(movie_name):
    """
    Fetch the poster URL for a given movie name using the TMDb API.
    """
    search_url = f'{BASE_URL}/search/movie'
    params = {
        'api_key': API_KEY,
        'query': movie_name
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        results = response.json().get('results', [])
        if results:
            poster_path = results[0].get('poster_path', '')
            if poster_path:
                return f'{IMAGE_BASE_URL}{poster_path}'  # Return full URL for the poster
    return None  # Return None if no poster is found

# Load the dataset
try:
    df = pd.read_csv('models/imdb_top_1000.csv')  # Path to your dataset
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
except FileNotFoundError:
    print("Error: Dataset file not found!")
    df = None

# Preprocess the dataset if it's loaded successfully
if df is not None:
    df = df.dropna(subset=['Overview'])  # Drop rows with missing descriptions
    df['Overview'] = df['Overview'].str.lower()  # Convert descriptions to lowercase

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english')  # Remove English stop words
    tfidf_matrix = tfidf.fit_transform(df['Overview'])  # Fit and transform descriptions

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


@app.route('/')
def home():
    """
    Render the homepage.
    """
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Handle movie recommendations and fetch posters dynamically using TMDb API.
    """
    if df is None:
        return jsonify({"error": "Dataset is not available. Please check your data file."})

    # Get the movie name from the form
    movie_name = request.form.get('search_query', '').lower()

    # Find the movie index
    try:
        movie_idx = df[df['Series_Title'].str.lower() == movie_name].index[0]
    except IndexError:
        return jsonify({"error": "Movie not found! Please try another title."})

    # Get similarity scores
    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get top 15 similar movies
    top_movies = similarity_scores[1:16]  # Skip the first one (itself)
    recommendations = []
    for i in top_movies:
        title = df.iloc[i[0]]['Series_Title']
        overview = df.iloc[i[0]]['Overview']

        # Fetch the high-quality poster URL using TMDb API
        poster_url = get_movie_poster(title) or df.iloc[i[0]]['Poster_Link']  # Fallback to dataset poster link if API fails

        recommendations.append({
            "title": title,
            "overview": overview,
            "poster": poster_url
        })

    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)
