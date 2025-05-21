import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_processed_data(path: str):
    return pd.read_csv(path)


def build_tfidf(corpus: pd.Series,
                max_features: int = 20000,
                ngram_range: tuple = (1, 2),
                stop_words: str = 'english'):
    """
    Fit TF-IDF on text series, return vectorizer and feature matrix.
    """
    vect = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
        strip_accents='unicode',

    )
    X = vect.fit_transform(corpus)
    return vect, X


# 3. Create feature matrix for baseline models
def create_feature_matrix(df: pd.DataFrame):
    """
    Constructs TF-IDF feature matrix and label vector for baseline classifiers.

    Returns:
        X: TF-IDF feature matrix
        y: target labels (Series)
        vect: fitted TfidfVectorizer
    """
    y = df['Genre'] if 'Genre' in df.columns else None
    vect, X = build_tfidf(df['Description_clean'])
    return X, y, vect
