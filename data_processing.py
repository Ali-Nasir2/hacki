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
        print("file not found.")
        return None




def clean_data(df):
    if df is None:
        print("Shit didnt get real as dataset didnt find")
        return None

    if 'Series_title' in df.columns:
        df['year']

if __name__ == "__main__":

