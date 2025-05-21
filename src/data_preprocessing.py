import os
import re
import unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split

# File paths and constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'raw'))
PROCESSED_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed'))
TRAIN_FILE = os.path.join(DATA_DIR, "train_data.txt")
TEST_FILE = os.path.join(DATA_DIR, "test_data.txt")
OTHER_THRESHOLD = 500
RANDOM_SEED = 2025
VAL_SPLIT = 0.1

os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_data(file_path, has_genre=True):
    if has_genre:
        columns = ['ID', 'Title', 'Genre', 'Description']
    else:
        columns = ['ID', 'Title', 'Description']

    data = pd.read_csv(
        file_path,
        sep=' ::: ',
        engine='python',
        names=columns,
        header=None,
        encoding='utf-8'
    )
    return data


def clean_text(text: str):
    """
    Apply light cleaning to text
    """
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def map_to_other(data: pd.DataFrame, threshold: int = OTHER_THRESHOLD):
    """
    Consolidate low frequency genres into 'Other'.
    """
    genre_counts = data['Genre'].value_counts()
    rare_genres = genre_counts[genre_counts < threshold].index.tolist()
    data['Genre'] = data['Genre'].apply(lambda x: 'Other' if x in rare_genres else x)
    return data


def split_data(data: pd.DataFrame):
    """
    Perform a stratified train-validation split.
    """
    train_df, val_df = train_test_split(
        data,
        test_size=VAL_SPLIT,
        stratify=data['Genre'],
        random_state=RANDOM_SEED
    )
    return train_df, val_df


def preprocess_data(data: pd.DataFrame, has_genre=True):
    """
    Clean text and optionally consolidate rare genres.
    """
    data['Description_clean'] = data['Description'].apply(clean_text)

    if has_genre:
        data = map_to_other(data)

    columns_to_keep = ['ID', 'Title', 'Description_clean']
    if has_genre:
        columns_to_keep.append('Genre')

    return data[columns_to_keep]


def save_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Save train, validation, and test sets.
    """
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val_processed.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"), index=False)


def main():
    print("Loading data...")
    train_data = load_data(TRAIN_FILE, has_genre=True)
    test_data = load_data(TEST_FILE, has_genre=False)

    print("Preprocessing data...")
    train_data = preprocess_data(train_data, has_genre=True)
    test_data = preprocess_data(test_data, has_genre=False)

    print("Splitting data...")
    train_df, val_df = split_data(train_data)

    print("Saving processed data...")
    save_data(train_df, val_df, test_data)

    print("Data preprocessing completed successfully.")


if __name__ == "__main__":
    main()
