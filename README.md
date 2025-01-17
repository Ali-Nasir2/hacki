FOR APP.PY
-
We imported flask and render_template to populate html dynamic pages. Request is used for handling user requests and jsonify is used to create handle and potray data in json format.

We import panda to read, clean and use the data properly from the csv file.

We import TfidfVectorizer from sklearn.feature_extraction.text, so that the data that is in the csv file can be used in numerical computations. Most imp is the TF-IDF, the Term Frequency-Inverse Document Frequency that assigns weight to the words in a document or in our case into the overview of the movies to further deduct a weight of them.

We import cosine_similarity from sklearn.metrics.pairwise in order to compute the similarity between the weights of the two movies that is deducted from them using their genre, overview etc. This draws an angle for them to see how similar the angle of two movies is.

We import ThreadPoolExecutor from concurrent.futures in order to run several task all together to save time in processing.


CODE STARTS
-

    API_KEY = ''
    BASE_URL = 'https://api.themoviedb.org/3'
    IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'

My Api key along with the Base Url of the API i am calling and the image base url is from where the images will be extracted and the image base url is given to tell the image taking path along the resolution of the image that will be displayed.

    executor = ThreadPoolExecutor(max_workers=10)

In here 10 worker threads are made to run the process of fetching data and do processes on the csv file faster.

    def get_movie_poster(movie_name):
    search_url = f'{BASE_URL}/search/movie' 

Movie is send from the html code through the api and then the poster is fetched.

    params = {'api_key': API_KEY, 'query': movie_name}
    response = requests.get(search_url, params=params)

Parameters are passed to the api key and a response is then requested.
 

        if response.status_code == 200:
        results = response.json().get('results', [])
        if results:
            poster_path = results[0].get('poster_path', '')
            if poster_path:
                return f'{IMAGE_BASE_URL}{poster_path}'  # Return full URL for the poster
    return None

If request is accepted then and response is given as OK then the results are extracted into a result array/list.
If the results do get stored in the list then start looking movie by movie and if the poster path exists for it then send its url-poster path that will be displayed on the page.

"results": [
        {
            "title": "Inception",
            "poster_path": "/qwe123.jpg"
        },
This is an example results list having poster path in it.
so the full url will be https://image.tmdb.org/t/p/w500/qwe123.jpg


    try:
    df = pd.read_csv('models/imdb_top_1000.csv')  # Path to your dataset
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

Try to load the dataset from the models directory and display message if it gets loaded successfully.

       df = df.dropna(subset=['Overview', 'Star1', 'Star2', 'Genre'])
df.dropna drops any of the columns where any of the above values was null to pre-reduce the dataset before any further calculations.
(df means dataframe in pandas)

    df['Overview'] = df['Overview'].str.lower()
    df['Genre'] = df['Genre'].str.lower()
 Converting the strings to lowercase to help in future analysis.

     df['Combined_Features'] = (
        df['Series_Title'].str.lower() + ' ' +
        df['Overview'] + ' ' +
        df['Star1'].str.lower() + ' ' +
        df['Star2'].str.lower() + ' ' +
        df['Genre']
    )

Makes an extra column in the dataset representing the combined values of the above inputs used, for example

Series_Title,	Overview,	Star1,	Star2,	Genre

Inception,	A mind-bending thriller,	Leonardo, Joseph,	Sci-Fi

gives the dataset value to be

Series_Title	Combined_Features

Inception---inception a mind-bending thriller leonardo joseph sci-fi


         tfidf = TfidfVectorizer(stop_words='english')
 Creates a vector and removes the unimportant words in english like,the,as,for



