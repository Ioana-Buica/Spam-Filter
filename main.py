import argparse
import os
import json
import re
import chardet

from bs4 import BeautifulSoup


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


def load_model_data(model_file="model_data.json"):
    """
    Get the values for class_probs, word_probs, vocab from file "model_data.json".
    """
    try:
        with open(model_file, "r") as f:
            model_data = json.load(f)

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
    tokens = preprocess_email_body(text)

    # Initialize scores with class probabilities
    scores = {cls: class_probs[cls] for cls in class_probs}

    # Iterate over the tokens/words and update the scores for each class
    for cls in class_probs:
        for word in tokens:
            if word in vocab:  # Only process words in the vocabulary
                # Add the word probability to the class score, using .get() to handle missing words
                scores[cls] += word_probs[cls].get(
                    word, 0
                )  # Default to 0 if the word is missing in word_probs

    if not scores:
        return "Score is empty"

    # Find the class with the maximum score
    return max(
        scores, key=scores.get, default="Unknown"
    )  # 'Unknown' if scores are somehow invalid


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


def process_data(data, item, file, class_probs, word_probs, vocab):
    if data.strip():
        prediction = predict(data, class_probs, word_probs, vocab)
        category = "inf" if prediction == "Spam" else "cln"
        file.write(f"{item}|{category}\n")


def process_file(item_path, item, output_file, class_probs, word_probs, vocab):
    with open(output_file, "a", encoding="utf-8") as file:
        try:
            encoding = detect_encoding(item_path)
            try:
                with open(item_path, "r", encoding=encoding, errors="ignore") as f:
                    process_data(f.read(), item, file, class_probs, word_probs, vocab)
                    
            except UnicodeDecodeError:
                with open(item_path, "r", encoding="utf-8", errors="replace") as f:
                    process_data(f.read(), item, file, class_probs, word_probs, vocab)
                    
        except FileNotFoundError:
            print(f"File not found: {item_path}")
        except PermissionError:
            print(f"Permission denied: {item_path}")
        except Exception as e:
            print(f"Error processing file '{item_path}': {e}")


def process_folder_with_subfolder(folder, output_file, class_probs, word_probs, vocab):
    """
    Scan the folder, process the files,
    and write in output_file if the file is inf or cln
    """
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)

        if os.path.isfile(item_path):
            process_file(
                item_path,
                item,
                output_file,
                class_probs,
                word_probs,
                vocab,
            )

        elif os.path.isdir(item_path):
            print(f"Processing subfolder: {item_path}")

            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)

                if os.path.isfile(sub_item_path):
                    process_file(
                        sub_item_path,
                        sub_item,
                        output_file,
                        class_probs,
                        word_probs,
                        vocab,
                    )


def scan_folder(folder, output_file):
    """
    Scan a folder with email files and classify and process them
    """
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return

    class_probs, word_probs, vocab = load_model_data("model_data.json")
    process_folder_with_subfolder(folder, output_file, class_probs, word_probs, vocab)
    print(f"Classification results written to {output_file}.")


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
