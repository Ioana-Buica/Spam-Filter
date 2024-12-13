import argparse
import os
import json
import re

import chardet


def preprocess(text):
    """
    Preprocess the text: lowercase, remove punctuation.
    Return a list of words.
    """
    text = text.lower()
    text = re.sub(r"\b\d+\b", "number", text)  # Normalize Numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove Punctuation and Non-words
    text = re.sub(r"\s+", " ", text).strip()  # Normalize Whitespace
    words = [word for word in text.split()]  # Tokenize
    return words


def load_model_data(model_file="model_data.json"):
    try:
        # Open and load the JSON file
        with open(model_file, "r") as f:
            model_data = json.load(f)

        # Extract the relevant parts of the model data
        class_probs = model_data.get("class_probs", {})
        word_probs = model_data.get("word_probs", {})
        vocab = model_data.get("vocab", [])

        return class_probs, word_probs, vocab
    except Exception as e:
        print(f"Error loading model data from {model_file}: {e}")
        return None, None, None


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


# Function to detect file encoding
def detect_encoding(file_path):
    """
    Detect the encoding of a file using chardet.
    """
    with open(file_path, "rb") as f:
        raw_data = f.read()  # Read the file as binary
    result = chardet.detect(raw_data)  # Detect the encoding
    return result["encoding"]


def scan_folder(folder, output_file):
    """
    Scan a folder for email files and classify them as spam or ham.
    """
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return

    class_probs, word_probs, vocab = load_model_data("model_data.json")

    try:
        with open(output_file, "w", encoding="utf-8") as file:
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)

                if os.path.isfile(item_path):
                    try:
                        # Detect file encoding
                        encoding = detect_encoding(item_path)

                        # Try opening the file with the detected encoding
                        try:
                            with open(
                                item_path, "r", encoding=encoding, errors="ignore"
                            ) as f:
                                data = f.read()

                                if data.strip():  # Skip empty files
                                    prediction = predict(
                                        data, class_probs, word_probs, vocab
                                    )
                                    category = "inf" if prediction == "Spam" else "cln"
                                    file.write(f"{item}|{category}\n")

                        except UnicodeDecodeError:
                            # If the detected encoding fails, try opening with a fallback encoding (utf-8)
                            print(
                                f"Warning: Unicode decode error with {item_path}. Trying UTF-8 encoding."
                            )
                            with open(
                                item_path, "r", encoding="utf-8", errors="ignore"
                            ) as f:
                                data = f.read()

                                if data.strip():  # Skip empty files
                                    prediction = predict(
                                        data, class_probs, word_probs, vocab
                                    )
                                    category = "inf" if prediction == "Spam" else "cln"
                                    file.write(f"{item}|{category}\n")

                    except Exception as e:
                        print(f"Error processing file '{item_path}': {e}")

            print(f"Classification results written to {output_file}.")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")


def write_info(output_file):
    """
    Writes Project_name, Student_name, Alias_Student, Project_version in the output file.
    """
    try:
        with open(output_file, "w") as f:
            Project_name = "SSOS"
            Student_name = "Buica Ioana-Alexandra"
            Alias_Student = "IO"
            Project_version = "1"

            f.write(
                f"Project name: {Project_name}\nStudent name: {Student_name}\nAlias Student: {Alias_Student}\nProject version: {Project_version}\n"
            )

        print(f"Information written to {output_file}.")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Anti-spam filter.")
    parser.add_argument(
        "-info",
        metavar="output_file",
        type=str,
        help="Write output file name where to be written: Project_name, Student_name, Alias_Student, Project_version.",
    )
    parser.add_argument(
        "-scan",
        nargs=2,
        metavar=("folder", "output_file"),
        help="Scan the folder and filter the files.",
    )

    args = parser.parse_args()

    if args.info:
        write_info(args.info)
    elif args.scan:
        scan_folder(args.scan[0], args.scan[1])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
