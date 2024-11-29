import argparse
import os
import json
import re
from collections import defaultdict
from math import log
import chardet

import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are available
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def preprocess(text):
    """
    Preprocess the text: lowercase, remove punctuation.
    Return a list of words.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    words = [word for word in text.split() if word not in stop_words]
    return words


def create_vocab(data):
    """
    Generate the vocabulary from the dataset, compute 
    word counts per label, class_counts, and class_totals
    """
    vocab = set()
    word_totals = defaultdict(lambda: defaultdict(int))
    class_counts = defaultdict(int)  # Count of documents per class
    class_totals = defaultdict(int)  # Total number of words per class

    for entry in data:
        if not isinstance(entry, dict) or "text" not in entry or "label" not in entry:
            print(f"Warning: Skipping invalid entry: {entry}")
            continue

        # Ensure 'text' is a list of tokens
        if isinstance(entry["text"], list):
            tokens = entry["text"]
        else:
            print(f"Warning: 'text' field is not a list in entry: {entry}")
            continue

        vocab.update(tokens)  # Add new words to the vocabulary

        # Get the label and update counts
        label = entry["label"]
        class_counts[label] += 1  # Increment document count for the class
        class_totals[label] += len(tokens)  # Increment word count for the class

        for token in tokens:
            word_totals[label][token] += (
                1  # Increment word count for the token in the class
            )

    return sorted(vocab), word_totals, class_counts, class_totals


def train_naive_bayes(word_totals, vocab, class_counts, class_totals):
    """
    Train a Naive Bayes model using the features and vocabulary.
    """

    class_probs = {
        cls: log(count / sum(class_counts.values()))
        for cls, count in class_counts.items()
    }

    word_probs = {
        cls: {
            word: log((word_totals[cls][word] + 1) / (class_totals[cls] + len(vocab)))
            for word in vocab
        }
        for cls in class_counts
    }

    return class_probs, word_probs


def predict(text, class_probs, word_probs, vocab):
    """
    Predict the class of a given text using the trained Naive Bayes model.
    """
    # Preprocess the input text 
    tokens = preprocess(text)

    # Initialize scores with class probabilities
    scores = {cls: class_probs[cls] for cls in class_probs}
    
    # Iterate over the tokens and update the scores for each class
    for cls in class_probs:
        for word in tokens:
            if word in vocab:  # Only process words in the vocabulary
                # Add the word probability to the class score, using .get() to handle missing words
                scores[cls] += word_probs[cls].get(
                    word, 0
                )  # Default to 0 if the word is missing in word_probs

    # If scores is empty or has only one class, we return it directly
    if not scores:
        return "Score is empty"  # Return a default value or handle it as needed

    # Find the class with the maximum score
    return max(
        scores, key=scores.get, default="Unknown"
    )  # 'Unknown' if scores are somehow invalid


def load_training_data(file_path):
    """
    Load training data from a JSON file where each entry has:
    - "text" as a list of tokens.
    - "label" as a classification (e.g., "Spam" or "Ham").
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Load the JSON data
        return data

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error: File '{file_path}' is not a valid JSON file. {e}")
    except Exception as e:
        print(f"Error reading training data: {e}")
    return []


def scan_folder(folder, output_file):
    """
    Scan a folder for email files and classify them as spam or ham.
    """
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return

    training_data = load_training_data("training_data.json")
    if not training_data:
        print("Error: No training data found. Cannot classify files.")
        return
    else:
        print("Training data found.")

    vocab, word_totals, class_counts, class_totals = create_vocab(
        training_data
    )
    if not vocab or not word_totals:
        print("Error: Vocabulary or word_totals are empty. Cannot train model.")
        return
    else:
        print("Vocabulary and word_totals are not empty.")

    if not class_counts or not class_totals:
        print("Error: class_counts or class_totals are empty. Cannot train model.")
        return
    else:
        print("class_counts and class_totals are not empty.")

    class_probs, word_probs = train_naive_bayes(
        word_totals, vocab, class_counts, class_totals
    )
    if not class_probs or not word_probs:
        print("Error: class_probs or word_totals are empty. Cannot train model.")
        return
    else:
        print("class_probs and word_totals are not empty.")

    try:
        with open(output_file, "w", encoding="utf-8") as file:
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)

                if os.path.isfile(item_path):
                    try:
                        # Detect the encoding of the file
                        with open(
                            item_path, "rb"
                        ) as f:  # Open the file in binary mode for chardet
                            raw_data = f.read()
                            result = chardet.detect(raw_data)
                            encoding = result["encoding"]  # Get the detected encoding

                        # Open the file with the detected encoding
                        with open(item_path, "r", encoding=encoding) as f:
                            data = f.read()

                            if data.strip():  # Skip empty files
                                prediction = predict(
                                    data, class_probs, word_probs, vocab
                                )
                                category = "inf" if prediction == "Spam" else "cln"
                                file.write(f"{item}|{category}\n")

                                print(f"File: {item}, Prediction: {prediction}")
                    except Exception as e:
                        print(f"Error processing file '{item}': {e}")

        print(f"Classification results written to {output_file}.")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Anti-spam filter.")
    parser.add_argument(
        "-scan",
        nargs=2,
        metavar=("folder", "output_file"),
        help="Scan the folder and classify files as spam or ham.",
    )

    args = parser.parse_args()

    if args.scan:
        scan_folder(args.scan[0], args.scan[1])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
