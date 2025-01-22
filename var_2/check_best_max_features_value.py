import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import time

from functions_load_data import load_data_from_folder

def train_and_evaluate_model(test_size=0.2):
    """
    Train and evaluate a spam filter with varying max_features values for TF-IDF.
    """
    try:
        spam_emails = load_data_from_folder("spam_path", 1)
        clean_emails = load_data_from_folder("clean_path", 0)

        if not spam_emails or not clean_emails:
            raise ValueError("Insufficient data for training. Check spam and clean folders.")

        all_emails = spam_emails + clean_emails
        max_features_list = [4000, 5000, 6000, 7000, 8000, 9000, 10000]

        for max_features in max_features_list:
            print(f"\nEvaluating with max_features={max_features}...")

            random.shuffle(all_emails)
            texts, labels = zip(*all_emails)

            X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=42)

            vectorizer = TfidfVectorizer(max_features=max_features)
            X_train_vec = vectorizer.fit_transform(X_train)

            model = SVC(C=1, gamma='scale', kernel='rbf')
            model.fit(X_train_vec, y_train)

            start_time = time.time() 
            X_test_vec = vectorizer.transform(X_test)
            y_pred = model.predict(X_test_vec)

            accuracy = accuracy_score(y_test, y_pred)

            end_time = time.time()  
            execution_time = end_time - start_time  

            print(f"max_features={max_features}: Accuracy = {accuracy * 100:.2f}%, Time taken = {execution_time:.4f} seconds")
        return model, vectorizer
    except Exception as e:
        print(f"Error in train_and_evaluate_model: {e}")
        return None, None
    
def main():
    model, vectorizer = train_and_evaluate_model()
    if model and vectorizer:
        print("\nModel trained and saved successfully!")
    else:
        print("Model training failed. Please check the data and try again.")

if __name__ == "__main__":
    main()
