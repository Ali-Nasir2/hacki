from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time
from concurrent.futures import ThreadPoolExecutor

# Initialize Flask app
app = Flask(__name__)

# TMDb API details
API_KEY = '8c4019d99263475e2689b2e90d461b78'
BASE_URL = 'https://api.themoviedb.org/3'
IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'  # Use medium resolution for faster loading

# Thread pool for fetching posters concurrently
executor = ThreadPoolExecutor(max_workers=10)


def get_movie_poster(movie_name):
    """
    Fetch the poster URL for a given movie name using the TMDb API.
    """
    search_url = f'{BASE_URL}/search/movie'
    params = {'api_key': API_KEY, 'query': movie_name}
    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        results = response.json().get('results', [])
        if results:
            poster_path = results[0].get('poster_path', '')
            if poster_path:
                return f'{IMAGE_BASE_URL}{poster_path}'  # Return full URL for the poster
    return None  # Return None if no poster is found


# Load and preprocess the dataset
try:
    start_time = time.time()
    df = pd.read_csv('models/imdb_top_1000.csv')  # Path to your dataset
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Drop rows with missing descriptions and preprocess the overview column
    df = df.dropna(subset=['Overview'])
    df['Overview'] = df['Overview'].str.lower()

    # Precompute TF-IDF vectors and similarity matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Overview'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Save processing time for debugging
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds.")

except FileNotFoundError:
    print("Error: Dataset file not found!")
    df = None


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

    # Fetch posters concurrently
    recommendations = []
    def process_movie(i):
        title = df.iloc[i[0]]['Series_Title']
        overview = df.iloc[i[0]]['Overview']
        poster_url = get_movie_poster(title) or df.iloc[i[0]]['Poster_Link']
        return {"title": title, "overview": overview, "poster": poster_url}

    with ThreadPoolExecutor() as executor:
        recommendations = list(executor.map(process_movie, top_movies))

    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)
