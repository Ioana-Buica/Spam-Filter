import argparse
import os
import re
import chardet
import pickle
import gzip

from collections import defaultdict
from math import log

from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from langdetect import detect


def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


def get_stopwords_for_language(language_code):
    try:
        return set(stopwords.words(language_code))
    except:
        return (
            set()
        )  # Return an empty set if stopwords for the language are unavailable


def detect_encoding(file_path):
    """
    Detect the encoding of a file using chardet.
    """

    with open(file_path, "rb") as f:
        raw_data = f.read()  # Read the file as binary
    result = chardet.detect(raw_data)  # Detect the encoding
    return result["encoding"]


def is_html(text):
    """
    Check if the text is html.
    """
    soup = BeautifulSoup(text, "html.parser")
    return bool(soup.find())  # True if it finds HTML tags


def preprocess_email_body(email_body):
    """
    Extract the words from email/file.
    """
    if is_html(email_body):
        # Parse the HTML content
        soup = BeautifulSoup(email_body, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()

        # Extract text content from the HTML
        text = soup.get_text(separator=" ")

    else:
        # Assume plain text
        text = email_body

    language_code = detect_language(text)  # Detect the language of the text
    stop_words = get_stopwords_for_language(
        language_code
    )  # Get the stopwords for the detected language

    # General text preprocessing
    text = text.lower()  # Lowercase
    text = re.sub(r"\b\d+\b", "", text)  # Remove Numbers
    text = re.sub(
        r"[^\w\s]", "", text
    )  # Remove all characters that are not alphanumeric
    text = re.sub(r"\s+", " ", text).strip()  # Normalize Whitespace
    words = [word for word in text.split()]  # Tokenize
    # stemmed_words = [stemmer.stem(word) for word in words]  # Apply Stemming
    words = [
        word for word in words if word.lower() not in stop_words
    ]  # Remove stop_words

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


def scan_folder(folder):
    """
    Scan a folder for email files and preprocess them for spam/ham classification.
    Save results in a file.
    """
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return

    processed_data = []
    # Iterate over folders in the folder
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)

        if os.path.isdir(item_path):  # If it’s a subdirectory
            # Use the directory name as the label
            label = item

            # Iterate over files in the subdirectory
            for file_name in os.listdir(item_path):
                file_path = os.path.join(item_path, file_name)

                if os.path.isfile(file_path):  # Check if it's a file
                    try:
                        # Detect file encoding
                        encoding = detect_encoding(file_path)

                        # Try opening the file with the detected encoding
                        try:
                            with open(
                                file_path, "r", encoding=encoding, errors="ignore"
                            ) as f:
                                data = f.read()

                                if data.strip():  # Ensure file is not empty
                                    processed_text = preprocess_email_body(data)
                                    processed_data.append(
                                        {"text": processed_text, "label": label}
                                    )
                        except UnicodeDecodeError:
                            # If the detected encoding fails, try opening with a fallback encoding (utf-8)
                            print(
                                f"Warning: Unicode decode error with {file_name}. Trying UTF-8 encoding."
                            )
                            with open(
                                file_path, "r", encoding="utf-8", errors="ignore"
                            ) as f:
                                data = f.read()

                                if data.strip():  # Ensure file is not empty
                                    processed_text = preprocess_email_body(data)
                                    processed_data.append(
                                        {"text": processed_text, "label": label}
                                    )
                    except Exception as e:
                        print(f"Error processing file '{file_name}': {e}")

    return processed_data


def save_model_data_compressed(model_data, filename):
    """
    Save the model data to a compressed file using pickle and gzip.
    """
    try:
        with gzip.open(filename, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model data saved to {filename}.")
    except Exception as e:
        print(f"Error saving model data: {e}")


def main():
    parser = argparse.ArgumentParser(description="Anti-spam filter.")
    parser.add_argument(
        "-scan",
        nargs=1,
        metavar=("folder"),
        help="Scan the folder, preprocess the files, and create a file with vocab.",
    )

    args = parser.parse_args()

    if args.scan:
        processed_data = scan_folder(args.scan[0])

        vocab, word_totals, class_counts, class_totals = create_vocab(processed_data)
        class_probs, word_probs = train_naive_bayes(
            word_totals, vocab, class_counts, class_totals
        )

        output_model_file = "model_data.pkl.gz"
        model_data = {
            "class_probs": class_probs,
            "word_probs": word_probs,
            "vocab": vocab,
        }

        save_model_data_compressed(model_data, output_model_file)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
