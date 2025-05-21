import joblib
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from feature_engineering import create_feature_matrix, load_processed_data


# 1. Load data & create TF–IDF features
def load_data_and_features(train_path, val_path):
    df_train = load_processed_data(train_path)
    df_val = load_processed_data(val_path)

    # create_feature_matrix now only takes the DataFrame
    X_train, y_train, vect = create_feature_matrix(df_train)

    # apply the same vectorizer to validation set
    X_val = vect.transform(df_val['Description_clean'])
    y_val = df_val['Genre']

    return X_train, y_train, X_val, y_val, vect


# 2. Training
def train_nb(X, y):
    return MultinomialNB().fit(X, y)


def train_lr(X, y, grid_search=True):
    if grid_search:
        # Define grid of hyperparameters to search over
        param_grid = {'C': [0.1, 1, 10]}
        # Initialize Logistic Regression with balanced class weights
        lr = LogisticRegression(class_weight='balanced', max_iter=1000)
        # Perform grid search with cross-validation to find best C based on f1_macro score
        grid = GridSearchCV(lr, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        grid.fit(X, y)
        return grid.best_estimator_
    # Train with default C=1 if grid search is not used
    return LogisticRegression(C=1, class_weight='balanced', max_iter=1000).fit(X, y)


def train_svm(X, y, grid_search=True):
    if grid_search:
        # Define grid of hyperparameters to search over
        param_grid = {'C': [0.1, 1, 10]}
        # Initialize Linear SVM with balanced class weights
        svc = LinearSVC(class_weight='balanced', max_iter=5000)
        # Perform grid search with cross-validation to find best C based on f1_macro score
        grid = GridSearchCV(svc, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        grid.fit(X, y)
        return grid.best_estimator_
    # Train with default C=1 if grid search is not used
    return LinearSVC(C=1, class_weight='balanced', max_iter=5000).fit(X, y)


# 3. Evaluation
def evaluate_model(model, X, y):
    preds = model.predict(X)

    # Get label names in the correct order
    labels = model.classes_
    target_names = [str(label) for label in labels]

    # Print classification report and confusion with class labels
    print(classification_report(y, preds, labels=labels, target_names=target_names))
    cm = confusion_matrix(y, preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print("Confusion Matrix:\n", cm_df.to_string())


def main():
    train_path = '../data/processed/train_processed.csv'
    val_path = '../data/processed/val_processed.csv'

    X_train, y_train, X_val, y_val, vect = load_data_and_features(train_path, val_path)

    # Baselines
    nb = train_nb(X_train, y_train)
    lr = train_lr(X_train, y_train)
    svm = train_svm(X_train, y_train)

    # Evaluate
    print("Naive Bayes:\n")
    evaluate_model(nb, X_val, y_val)

    print("Logistic Regression:\n")
    evaluate_model(lr, X_val, y_val)

    print("Linear SVM:\n")
    evaluate_model(svm, X_val, y_val)


if __name__ == '__main__':
    main()
