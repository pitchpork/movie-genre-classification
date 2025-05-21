# 🎬 Movie Genre Classification with BERT and Classical Models

This project explores the task of classifying movie plot summaries into genres using both classical machine learning techniques and transformer-based models. We finetune BERT and compare its performance to traditional baselines like Naive Bayes, Logistic Regression, and SVM.


---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/movie-genre-classification.git
cd movie-genre-classification
````

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

---

## 📊 Exploratory Data Analysis

Run the following script to generate visualizations of the dataset:

```bash
python eda/analysis.py
```

This will save:

* A bar chart showing the distribution of genre labels.
* A histogram showing the distribution of plot summary lengths.

---

## 🧹 Data Preprocessing

To perform light cleaning and stratified splitting of the dataset:

```bash
python src/data_preprocessing.py
```

This will create preprocessed datasets and save them under `data/processed/`.

---

## 📉 Run Classical Baselines

To train and evaluate classical models (Naive Bayes, Logistic Regression, Linear SVM):

```bash
python src/model_baseline.py
```

Performance metrics and confusion matrices will be printed to the console.

---

## 🤖 Run BERT Transformer Model

To fine-tune BERT on the genre classification task:

```bash
python src/model_transformer.py
```

> ⚠️ **Note:** Training BERT is computationally expensive. A CUDA-enabled GPU is **strongly recommended** for efficient training.

Trained model weights and training history will be saved in the `models/bert_classifier` directory.

---

## 📈 Model Evaluation

After training, you can evaluate the saved BERT model:

```bash
python src/evaluate.py
```

This will load the saved model and print the classification report and confusion matrix.

---
