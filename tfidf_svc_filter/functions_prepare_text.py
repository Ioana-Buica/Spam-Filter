import re
from bs4 import BeautifulSoup

from functions_stopwrods import remove_stopwords


def preprocess_email_body(email_body):
    """
    Extract and clean words from an email body.
    """
    try:
        text = email_body

        if "<" in text and ">" in text:
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text(separator="", strip=True)

        # Text preprocessing
        text = text.lower()  # Lowercase
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
        text = re.sub(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", text
        )  # Remove email addresses

        text = re.sub(r"\b\d+\b", "", text)  # Remove numbers
        text = re.sub(r"[^\w\s]", "", text)  # Remove non-alphanumeric characters

        text = remove_stopwords(text)
        text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
        return text

    except Exception as e:
        print(f"Error in preprocess_email_body: {e}")
        return ""
