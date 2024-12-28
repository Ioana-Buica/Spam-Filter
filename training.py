import argparse
import os
import re
import json
import chardet

from collections import defaultdict
from math import log
from bs4 import BeautifulSoup


def detect_encoding(file_path):
    """
    Detect the encoding of a file using chardet.
    """
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()

        result = chardet.detect(raw_data)
        return result["encoding"]

    except PermissionError:
        print(f"Error: Permission denied for file '{file_path}'.")
    except Exception as e:
        print(f"Error detecting encoding for file '{file_path}': {e}")


def is_html(text):
    try:
        soup = BeautifulSoup(text, "html.parser")
        return bool(soup.find())

    except Exception as e:
        print(f"Error parsing text as HTML: {e}")
        return False


def preprocess_email_body(email_body):
    if is_html(email_body):
        # Parse the HTML content
        soup = BeautifulSoup(email_body, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()

        # Extract text content from the HTML
        text = soup.get_text(separator=" ")

    text = email_body.lower()  # Convert to lowercase
    # text = re.sub(r"\b\d+\b", "number", text)  # Normalize numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    words = text.split()  # Tokenize text
    return words


def create_vocab(data):
    """
    Generate the vocabulary and compute word counts and totals for each label(Spam/Clean).
    """
    vocab = set()
    word_totals = defaultdict(lambda: defaultdict(int))
    class_counts = defaultdict(int)
    class_totals = defaultdict(int)

    for entry in data:
        if not isinstance(entry, dict) or "text" not in entry or "label" not in entry:
            print(f"Warning: Skipping invalid entry: {entry}")
            continue

        tokens = entry.get("text", [])
        label = entry["label"]

        vocab.update(tokens)
        class_counts[label] += 1
        class_totals[label] += len(tokens)

        for token in tokens:
            word_totals[label][token] += 1

    return sorted(vocab), word_totals, class_counts, class_totals


def train_naive_bayes(word_totals, vocab, class_counts, class_totals):
    """
    Train a Naive Bayes model using the given data.
    """
    try:
        class_probs = {
            cls: log(count / sum(class_counts.values()))
            for cls, count in class_counts.items()
        }

        word_probs = {
            cls: {
                word: log(
                    (word_totals[cls][word] + 1) / (class_totals[cls] + len(vocab))
                )
                for word in vocab
            }
            for cls in class_counts
        }

        return class_probs, word_probs
    except ZeroDivisionError:
        print("Error: Division by zero encountered in training. Check input data.")


def scan_folder(folder):
    """
    Scan a folder for email files and preprocess them for classification.
    """
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder '{folder}' does not exist.")

    processed_data = []
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)

        if os.path.isdir(item_path):
            label = item

            for file_name in os.listdir(item_path):
                file_path = os.path.join(item_path, file_name)

                if os.path.isfile(file_path):
                    try:
                        encoding = detect_encoding(file_path)

                        with open(
                            file_path, "r", encoding=encoding, errors="ignore"
                        ) as f:
                            data = f.read()

                            if data.strip():
                                processed_text = preprocess_email_body(data)
                                processed_data.append(
                                    {"text": processed_text, "label": label}
                                )

                    except Exception as e:
                        print(f"Error processing file '{file_name}': {e}")

    return processed_data


def main():
    parser = argparse.ArgumentParser(description="Anti-spam filter.")
    parser.add_argument(
        "-scan",
        nargs=1,
        metavar=("folder"),
        help="Scan the folder, preprocess, and save as JSON.",
    )

    args = parser.parse_args()

    if args.scan:
        folder = args.scan[0]
        processed_data = scan_folder(folder)

        vocab, word_totals, class_counts, class_totals = create_vocab(processed_data)
        class_probs, word_probs = train_naive_bayes(
            word_totals, vocab, class_counts, class_totals
        )

        output_model_file = "model_data.json"
        model_data = {
            "class_probs": class_probs,
            "word_probs": word_probs,
            "vocab": vocab,
        }

        try:
            with open(output_model_file, "w") as f:
                json.dump(model_data, f, indent=4)

            print(f"Model data saved to {output_model_file}.")

        except Exception as e:
            print(f"Error saving model data: {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
