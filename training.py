import argparse
import os
import re
import json

from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")  # Download the stopwords dataset if not already downloaded
stop_words = set(stopwords.words("english"))  # Load the English stopwords


def is_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return bool(soup.find())  # True if it finds HTML tags


def preprocess_email_body(email_body):
    if is_html(email_body):
        # Parse HTML and extract text
        soup = BeautifulSoup(email_body, "html.parser")
        text = soup.get_text(separator=" ")
        return []
    else:
        # Assume plain text
        text = email_body

    # General text preprocessing
    text = text.lower()  # Lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    words = [word for word in text.split() if word not in stop_words]
    return words


def scan_folder(folder, output_file="training_data.json"):
    """
    Scan a folder for email files and preprocess them for spam/ham classification.
    Save results in a JSON file.
    """
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return

    processed_data = []

    try:
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
                            with open(file_path, "r") as f:
                                data = f.read()

                                if data.strip():  # Ensure file is not empty
                                    # Preprocess the file content
                                    processed_text = preprocess_email_body(data)

                                    # Append the processed data
                                    processed_data.append(
                                        {"text": processed_text, "label": label}
                                    )
                        except Exception as e:
                            print(f"Error processing file '{file_name}': {e}")

    except Exception as e:
        print(f"Error reading folder '{folder}': {e}")

    # Save processed data to JSON
    try:
        with open(output_file, "w") as f:
            json.dump(processed_data, f, indent=4)
        print(f"Preprocessed data saved to {output_file}.")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")


def main():
    parser = argparse.ArgumentParser(description="Anti-spam filter.")
    parser.add_argument(
        "-scan",
        nargs=2,
        metavar=("folder", "output_file"),
        help="Scan the folder, preprocess, and save as JSON.",
    )

    args = parser.parse_args()

    if args.scan:
        scan_folder(args.scan[0])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
