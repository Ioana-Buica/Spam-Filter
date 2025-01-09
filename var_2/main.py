import os
import argparse
import re
import chardet
import joblib

from bs4 import BeautifulSoup


def detect_encoding(file_path):
    """
    Detect the encoding of a file using chardet.
    Default to UTF-8 if detection fails.
    """
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()
        result = chardet.detect(raw_data)
        return result["encoding"]
    except Exception as e:
        print(f"Error detecting encoding for file {file_path}: {e}")
        return "utf-8"


def is_html(content):
    """
    Determine if the email content is HTML.
    """
    try:
        return bool(BeautifulSoup(content, "html.parser").find())
    except Exception as e:
        print(f"Error in is_html: {e}")
        return False


def preprocess_email_body(email_body):
    """
    Extract and clean words from an email body.
    """
    try:
        if is_html(email_body):
            soup = BeautifulSoup(email_body, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
        else:
            text = email_body

        # Text preprocessing
        text = text.lower()  # Lowercase
        text = re.sub(r"\b\d+\b", "", text)  # Remove numbers
        text = re.sub(r"[^\w\s]", "", text)  # Remove non-alphanumeric characters
        text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace

        return text

    except Exception as e:
        print(f"Error in preprocess_email_body: {e}")
        return ""


def read_email_file(file_path, encoding):
    """
    Read the content of the email file with the given encoding.
    """
    try:
        with open(file_path, "r", encoding=encoding, errors="ignore") as file:
            lines = file.readlines()
            if not lines:
                return None

            subject = lines[0].strip()
            content = "".join(lines[1:])
            email_text = subject + " " + content

            return email_text
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def classify_email(email_text, model, vectorizer):
    """
    Classify a single email and return the verdict (inf or cln).
    """
    processed_text = preprocess_email_body(email_text)

    if processed_text:
        text_vector = vectorizer.transform([processed_text]).toarray()
        prediction = model.predict(text_vector)
        return "inf" if prediction == 1 else "cln"

    return None


def write_classification_result(file_name, verdict, out_file):
    """
    Write the classification result to the output file.
    """
    formatted_line = f"{file_name.strip()}|{verdict}\n"
    out_file.write(formatted_line)


def process_files_in_folder(folder_path, model, vectorizer, output_file):
    """
    Process all email files in the given folder (non-recursively),
    classify them, and write the result in output_file.
    """
    try:
        with open(output_file, "w") as out_file:
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if os.path.isdir(file_path):
                    for sub_item in os.listdir(file_path):
                        sub_item_path = os.path.join(file_path, sub_item)
                        if os.path.isfile(sub_item_path):
                            encoding = detect_encoding(sub_item_path)
                            email_text = read_email_file(sub_item_path, encoding)
                            if email_text:
                                verdict = classify_email(email_text, model, vectorizer)
                                if verdict:
                                    write_classification_result(
                                        sub_item, verdict, out_file
                                    )

                if os.path.isfile(file_path):
                    encoding = detect_encoding(file_path)
                    email_text = read_email_file(file_path, encoding)
                    if email_text:
                        verdict = classify_email(email_text, model, vectorizer)
                        if verdict:
                            write_classification_result(file_name, verdict, out_file)

    except Exception as e:
        print(f"Error processing folder '{folder_path}': {e}")


def write_info(output_file):
    """
    Writes Project_name, Student_name, Alias_Student, Project_version in the output file.
    """
    try:
        with open(output_file, "w") as out_file:
            out_file.write("SSOS\n")
            out_file.write("Buica Ioana-Alexandra\n")
            out_file.write("IO\n")
            out_file.write("1.0\n")
        print(f"Information written to {output_file}")

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
        help="Scan the folder and write in output_file if the emails are spam or clean.",
    )

    args = parser.parse_args()
    try:
        # Load the trained model and vectorizer from the saved files
        model = joblib.load("spam_filter_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")

        # Handle the "info" argument
        if args.info:
            write_info(args.info)

        # Handle the "scan" argument
        elif args.scan:
            folder = args.scan[0]
            output_file = args.scan[1]

            if os.path.isdir(folder):
                process_files_in_folder(folder, model, vectorizer, output_file)
                print(f"Predictions written to {output_file}")
            else:
                print(
                    f"Error: The folder '{folder}' does not exist or is not a directory."
                )

        # If no arguments were provided, show help
        else:
            parser.print_help()

    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")


if __name__ == "__main__":
    main()
