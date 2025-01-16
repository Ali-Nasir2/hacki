import pandas as pd


def load_data():

    file_path = 'models/imdb_top_1000.csv'
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
        print("Sample Data:")
        print(df.head())
        return df
    except FileNotFoundError:
        print("Error: File not found. Please ensure the file path is correct.")
        return None


def clean_data(df):
    """
    Clean and preprocess the dataset.
    """
    if df is None:
        print("Dataset not found. Cannot clean data.")
        return None

    if 'Series_Title' not in df.columns or 'Overview' not in df.columns:
        print("Error: Missing required columns ('Series_Title', 'Overview').")
        return None


    df = df.dropna(subset=['Series_Title', 'Overview'])

    df['Overview'] = df['Overview'].str.lower()

    df['Series_Title'] = df['Series_Title'].str.strip()

    print("Data cleaning complete. Sample Data:")
    print(df.head())
    return df


if __name__ == "__main__":
    # Load the dataset
    data = load_data()

    # Clean the dataset
    cleaned_data = clean_data(data)

    if cleaned_data is not None:
        print(f"Cleaned dataset has {cleaned_data.shape[0]} rows and {cleaned_data.shape[1]} columns.")
