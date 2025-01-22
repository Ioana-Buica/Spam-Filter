import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from functions_load_data import load_data_from_folder


def train_and_evaluate_model():
    """
    Train a spam filter using data from spam and clean folders.
    """
    try:
        spam_emails = load_data_from_folder("spam_path", 1)
        clean_emails = load_data_from_folder("clean_path", 0)

        if not spam_emails or not clean_emails:
            raise ValueError(
                "Insufficient data for training. Check spam and clean folders."
            )

        all_emails = spam_emails + clean_emails
        texts, labels = zip(*all_emails)

        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(texts)
        y = labels

        model = SVC(C=1, gamma="scale", kernel="rbf")
        model.fit(X, y)

        joblib.dump(model, "spam_filter_model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")
        print("Model and vectorizer saved successfully!")
        return model, vectorizer
    except Exception as e:
        print(f"Error in train_and_evaluate_model: {e}")
        return None, None


if __name__ == "__main__":
    model, vectorizer = train_and_evaluate_model()
