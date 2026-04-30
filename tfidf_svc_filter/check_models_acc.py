import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from functions_load_data import load_data_from_folder

def train_and_evaluate_model(test_size=0.2):
    """
    Train and evaluate different models to determine the one with the best accuracy.
    """
    try:
        spam_emails = load_data_from_folder("spam_path", 1)
        clean_emails = load_data_from_folder("clean_path", 0)

        if not spam_emails or not clean_emails:
            raise ValueError("Insufficient data for training. Check spam and clean folders.")

        all_emails = spam_emails + clean_emails
        random.shuffle(all_emails)
        texts, labels = zip(*all_emails)

        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(texts)
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        models = {
            "MultinomialNB": {
                "model": MultinomialNB(),
                "params": {"alpha": [0.1, 0.5, 1.0]}
            },
            "SVC": {
                "model": SVC(),
                "params": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"], "kernel": ["linear", "rbf"]}
            },
            "RandomForestClassifier": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                }
            }
        }

        best_model = None
        best_accuracy = 0
        best_model_name = ""

        for model_name, model_info in models.items():
            print(f"\nTraining and tuning {model_name} with GridSearchCV...")

            # Set up GridSearchCV
            grid_search = GridSearchCV(
                estimator=model_info["model"],
                param_grid=model_info["params"],
                cv=3,
                scoring='accuracy',
                verbose=2,
                n_jobs=-1
            )

            try:
                grid_search.fit(X_train, y_train)
                best_grid_model = grid_search.best_estimator_
                y_pred = best_grid_model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                print(f"Best parameters for {model_name}: {grid_search.best_params_}")
                print(f"Accuracy of {model_name}: {accuracy:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = best_grid_model
                    best_model_name = model_name

                print(f"Classification Report for {model_name}:\n", classification_report(y_test, y_pred))
            except Exception as e:
                print(f"Error during GridSearchCV for {model_name}: {e}")

        print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
        return best_model, vectorizer

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