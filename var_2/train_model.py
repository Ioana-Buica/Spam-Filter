import os
import re
import chardet
import joblib

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


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
    Extract and clean words from an email body (tokenize without stopwords).
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

        # Tokenize text (split into words)
        # words = text.split()

        return text  # Return tokenized text (list of words)

    except Exception as e:
        print(f"Error in preprocess_email_body: {e}")
        return ""


def load_data_from_folder(folder_path, label):
    """
    Load text data from a folder and assign the given label.
    """
    data = []
    try:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                encoding = detect_encoding(file_path)
                try:
                    with open(
                        file_path, "r", encoding=encoding, errors="ignore"
                    ) as file:
                        lines = file.readlines()
                        if not lines:
                            continue

                        subject = lines[0].strip()
                        content = "".join(lines[1:])
                        email_text = subject + " " + content

                        processed_text = preprocess_email_body(email_text)
                        data.append((processed_text, label))

                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    except Exception as e:
        print(f"Error loading data from folder '{folder_path}': {e}")
    return data


def train_and_evaluate_model(spam_path, clean_path, test_size=0.2):
    """
    Train a spam filter using data from spam and clean folders.
    """
    try:
        spam_emails = load_data_from_folder(spam_path, 1)
        clean_emails = load_data_from_folder(clean_path, 0)

        if not spam_emails or not clean_emails:
            raise ValueError(
                "Insufficient data for training. Check spam and clean folders."
            )

        all_emails = spam_emails + clean_emails
        texts, labels = zip(*all_emails)

        # Vectorizer (no stopword filtering)
        vectorizer = TfidfVectorizer(max_features=2000)
        X = vectorizer.fit_transform(texts).toarray()  # X is the vectorized text
        y = labels  # y is the labels (0 or 1)

        # Train model
        model = MultinomialNB()
        model.fit(X, y)

        # Save model and vectorizer
        joblib.dump(model, "spam_filter_model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")
        print("Model and vectorizer saved successfully!")

        return model, vectorizer
    except Exception as e:
        print(f"Error in train_and_evaluate_model: {e}")
        return None, None


# Main function to train the model
def main():
    spam_folder = r"C:\Users\Io\Desktop\Lot1_\Lot1_\Lot1\Spam"
    clean_folder = r"C:\Users\Io\Desktop\Lot1_\Lot1_\Lot1\Clean"

    model, vectorizer = train_and_evaluate_model(spam_folder, clean_folder)


if __name__ == "__main__":
    main()
