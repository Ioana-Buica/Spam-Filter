# Spam Filter Repository

This repository contains two separate implementations of an email spam filter. Each folder is a different version of the project with its own preprocessing, training, and prediction pipeline.

## Folder Overview

### `naive_bayes_filter`

This folder contains a custom Naive Bayes spam classifier built with manual vocabulary generation and text preprocessing.

- `training.py`

  - Reads email files from labeled folders.
  - Detects file encoding, removes HTML tags, tokenizes text, and normalizes punctuation and numbers.
  - Builds a vocabulary and computes Naive Bayes class and word probabilities.
  - Saves the trained model data as `model_data.pkl.gz`.
- `main.py`

  - Loads the compressed Naive Bayes model data.
  - Scans a folder and its subfolders for email files.
  - Classifies each file as spam or clean.
  - Writes results to an output file with labels `inf` for spam and `cln` for clean.

### `tfidf_svc_filter`

This folder contains a more modern machine learning pipeline using TF-IDF vectorization and scikit-learn classifiers.

- `train_model.py`

  - Loads labeled email data from separate spam and clean directories.
  - Vectorizes email text using `TfidfVectorizer`.
  - Trains an `SVC` model and saves `spam_filter_model.pkl` and `vectorizer.pkl`.
- `main.py`

  - Loads the saved model and vectorizer.
  - Reads email files from a target folder.
  - Processes text and predicts spam vs clean in batch.
  - Writes output in the same `filename|label` format.
- `functions_load_data.py`

  - Loads email files from a folder and returns preprocessed text and labels.
  - Uses threading to speed up file loading.
- `functions_prepare_text.py`

  - Cleans email text by stripping HTML, removing URLs, email addresses, punctuation, and stopwords.
- `functions_stopwrods.py`

  - Detects the email language and removes language-specific stopwords using NLTK.
- `check_models_acc.py`

  - Trains and evaluates multiple classifiers with `GridSearchCV`.
  - Compares `MultinomialNB`, `SVC`, and `RandomForestClassifier`.
- `check_best_max_features_value.py`

  - Tests different TF-IDF `max_features` settings.
  - Prints accuracy and timing for each configuration.

## How to Use

### Run the Naive Bayes version (`naive_bayes_filter`)

1. Train the model using labeled data:
   ```bash
   python naive_bayes_filter/training.py -scan <labeled_data_folder>
   ```
2. Classify files in a folder:
   ```bash
   python naive_bayes_filter/main.py -scan <target_folder> <output_file>
   ```

### Run the TF-IDF + SVC version (`tfidf_svc_filter`)

1. Train the model and save artifacts:
   ```bash
   python tfidf_svc_filter/train_model.py
   ```
2. Classify files in a folder:
   ```bash
   python tfidf_svc_filter/main.py -scan <target_folder> <output_file>
   ```

> Note: `train_model.py` currently refers to placeholder folders `spam_path` and `clean_path`. Update those folder names in the script or rename your data directories accordingly.

## Output Format

Output files use this format:

```
<filename>|<label>
```

Where `<label>` is:

- `inf` — spam
- `cln` — clean

## Dependencies

The code uses the following Python libraries:

- `beautifulsoup4`
- `chardet`
- `nltk`
- `langdetect`
- `scikit-learn`
- `joblib`
- `bs4`

## What This Project Demonstrates

This repository showcases practical machine learning and natural language processing skills:

- **Text Preprocessing**: Handling HTML, encoding detection, tokenization, stopword removal, and language detection.
- **Classical ML**: Implementing Naive Bayes from scratch for text classification.
- **Modern ML Pipeline**: Using scikit-learn for vectorization, model training, and evaluation.
- **Model Comparison**: Hyperparameter tuning and comparing multiple classifiers.
- **Performance Optimization**: Threading for data loading and testing different feature settings.
- **Python Best Practices**: Modular code structure, error handling, and command-line interfaces.

It's a great example of building end-to-end ML solutions for real-world problems like email filtering.
This project implements an anti-spam filter that preprocesses email data, builds a vocabulary, and trains a Naive Bayes model for spam classification.
