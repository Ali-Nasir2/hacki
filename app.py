from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from concurrent.futures import ThreadPoolExecutor

# Initialize Flask app
app = Flask(__name__)

# TMDb API details
API_KEY = '8c4019d99263475e2689b2e90d461b78'
BASE_URL = 'https://api.themoviedb.org/3'
IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'

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
    df = pd.read_csv('models/imdb_top_1000.csv')  # Path to your dataset
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Drop rows with missing required data
    df = df.dropna(subset=['Overview', 'Star1', 'Star2', 'Genre'])
    df['Overview'] = df['Overview'].str.lower()
    df['Genre'] = df['Genre'].str.lower()

    # Combine features for similarity calculation
    df['Combined_Features'] = (
        df['Series_Title'].str.lower() + ' ' +
        df['Overview'] + ' ' +
        df['Star1'].str.lower() + ' ' +
        df['Star2'].str.lower() + ' ' +
        df['Genre']
    )

    # Precompute TF-IDF vectors and similarity matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Combined_Features'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

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

    # Title-Based Search
    title_matches = df[df['Series_Title'].str.lower().str.contains(movie_name, na=False)]
    recommendations = []
    if not title_matches.empty:
        for _, row in title_matches.iterrows():
            poster_url = get_movie_poster(row['Series_Title']) or row['Poster_Link']
            recommendations.append({
                "title": row['Series_Title'],
                "overview": row['Overview'],
                "poster": poster_url
            })

    # If fewer than 10 title matches, expand to feature-based search
    if len(recommendations) < 10:
        # Find the movie index
        try:
            movie_idx = df[df['Series_Title'].str.lower() == movie_name].index[0]
        except IndexError:
            return jsonify({"error": "Movie not found! Please try another title."})

        # Get similarity scores
        similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Add recommendations based on features
        for i in similarity_scores[1:21]:  # Skip the first one (itself)
            movie = df.iloc[i[0]]
            poster_url = get_movie_poster(movie['Series_Title']) or movie['Poster_Link']
            recommendations.append({
                "title": movie['Series_Title'],
                "overview": movie['Overview'],
                "poster": poster_url
            })

    # Limit to top 20 recommendations
    recommendations = recommendations[:20]

    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)
