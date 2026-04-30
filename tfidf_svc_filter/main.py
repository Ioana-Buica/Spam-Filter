import os
import argparse
import joblib
from joblib import Parallel, delayed

from functions_prepare_text import preprocess_email_body


def process_email(file_path, model, vectorizer):
    """
    Process the email, extract features, and predict whether it's spam or ham.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            email_text = file.read()
            if not email_text:
                return None

            processed_text = preprocess_email_body(email_text)
            if not processed_text:
                return None

            text_vector = vectorizer.transform([processed_text])
            prediction = model.predict(text_vector)
            verdict = "inf" if prediction == 1 else "cln"
            return f"{os.path.basename(file_path)}|{verdict}"
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def batch_predict(folder_path, model, vectorizer, output_file):
    all_files = []
    for item_name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item_name)

        if os.path.isfile(item_path):
            all_files.append(item_path)

        elif os.path.isdir(item_path):
            all_files.extend(
                os.path.join(item_path, f)
                for f in os.listdir(item_path)
                if os.path.isfile(os.path.join(item_path, f))
            )

    # or use Threads
    predictions = Parallel(n_jobs=-1)(
        delayed(process_email)(file_path, model, vectorizer) for file_path in all_files
    )

    with open(output_file, "w") as out_file:
        out_file.write("\n".join(filter(None, predictions)))


def main():
    parser = argparse.ArgumentParser(description="Anti-spam filter.")
    parser.add_argument(
        "-scan",
        nargs=2,
        metavar=("folder", "output_file"),
        help="Scan the folder and write in output_file if the emails are spam or clean.",
    )

    args = parser.parse_args()
    try:
        if args.scan:
            folder = args.scan[0]
            output_file = args.scan[1]

            if os.path.isdir(folder):
                model = joblib.load("spam_filter_model.pkl")
                vectorizer = joblib.load("vectorizer.pkl")
                batch_predict(folder, model, vectorizer, output_file)
                print(f"Predictions written to {output_file}")
            else:
                print(
                    f"Error: The folder '{folder}' does not exist or is not a directory."
                )

        else:
            parser.print_help()

    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")


if __name__ == "__main__":
    main()
