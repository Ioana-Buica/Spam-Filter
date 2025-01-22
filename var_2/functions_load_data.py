import os
import chardet
from concurrent.futures import ThreadPoolExecutor

from functions_prepare_text import preprocess_email_body


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


def read_email_file(file_path):
    """
    Read the content of the email file with the given encoding.
    """
    try:
        encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding=encoding, errors="ignore") as file:
            lines = file.readlines()
            if not lines:
                return None

            subject = lines[0].strip()
            body = "".join(lines[1:])
            email_text = subject + " " + body
            
            processed_text = preprocess_email_body(email_text)

            return processed_text

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def load_data_from_folder(folder_path, label):
    """
    Load text data from a folder and assign the given label.
    """
    data = []
    try:
        with os.scandir(folder_path) as entries:
            file_paths = [entry.path for entry in entries if entry.is_file()]

        def process_file(file_path):
            try:
                processed_text = read_email_file(file_path)
                return (processed_text, label)
            except Exception as e:
                print(f"Error processing file '{file_path}': {e}")
                return None

        with ThreadPoolExecutor() as executor:
            results = executor.map(process_file, file_paths)
            data.extend(filter(None, results))

    except Exception as e:
        print(f"Error loading data from folder '{folder_path}': {e}")

    return data
