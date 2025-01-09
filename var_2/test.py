import os
import re
import random
import chardet
import joblib

from bs4 import BeautifulSoup

from langdetect import detect, DetectorFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


# Ensure consistent language detection
DetectorFactory.seed = 0


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


def load_data_from_folder(folder_path, label):
    """
    Load text data from a folder and assign the given label.
    """
    data = []
    try:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                encoding = detect_encoding(file_path)  # Detect encoding
                try:
                    with open(
                        file_path, "r", encoding=encoding, errors="ignore"
                    ) as file:
                        lines = file.readlines()
                        if not lines:
                            continue  # Skip empty files
                        subject = lines[0].strip()  # First line is the subject
                        content = "".join(lines[1:])  # Combine the rest as content
                        email_text = (
                            subject + " " + content
                        )  # Combine subject and content
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

        # Combine and shuffle
        all_emails = spam_emails + clean_emails
        random.shuffle(all_emails)

        # Separate texts and labels
        texts, labels = zip(*all_emails)

        vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        X = vectorizer.fit_transform(texts).toarray()
        y = labels

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Train model
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # Save model and vectorizer
        joblib.dump(model, "spam_filter_model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")

        return model, vectorizer
    except Exception as e:
        print(f"Error in train_and_evaluate_model: {e}")
        return None, None


# Function: Predict spam
def predict_spam(email_text, model, vectorizer):
    """
    Predict whether an email is spam or not.
    """
    try:
        cleaned_text = preprocess_email_body(email_text)
        email_vectorized = vectorizer.transform([cleaned_text]).toarray()
        prediction = model.predict(email_vectorized)[0]
        return "Spam" if prediction == 1 else "Not Spam"
    except Exception as e:
        print(f"Error in predict_spam: {e}")
        return "Error"


# Main function to execute the program
def main():
    spam_folder = r"C:\Users\Io\Desktop\Lot1_\Lot1_\Lot1\Spam"
    clean_folder = r"C:\Users\Io\Desktop\Lot1_\Lot1_\Lot1\Clean"

    # Train and evaluate the model
    model, vectorizer = train_and_evaluate_model(spam_folder, clean_folder)

    if model and vectorizer:
        print("\nModel trained and saved successfully!")

        # Example email for prediction
        email_text = (
            "Subject: Win a prize\nBody: Click here to claim your $1,000 reward now!"
        )
        result = predict_spam(email_text, model, vectorizer)
        print(f"\nPredicted: {result}")
    else:
        print("Model training failed. Please check the data and try again.")


if __name__ == "__main__":
    main()
