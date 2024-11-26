import argparse
import os
import json
import re
from collections import defaultdict, Counter
from math import log


# Static training dataset
training_data = [
    ("Free money now!", "spam"),
    ("You have won a lottery, click to claim.", "spam"),
    ("Meeting at 10 AM in room 305.", "ham"),
    ("Don't forget to submit your assignment.", "ham"),
    ("Exclusive offer for you, win a prize.", "spam"),
    ("Let's catch up tomorrow.", "ham"),
]


# Step 1: Preprocessing
def preprocess(text):
    """
    Preprocess the text: lowercase, remove punctuation, and tokenize.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text.split()


# Step 2: Create Vocabulary and Feature Vectors
def create_vocab_and_features(data):
    """
    Generate the vocabulary and feature vectors from the dataset.
    """
    vocab = set()
    features = []

    for text, label in data:
        tokens = preprocess(text)
        vocab.update(tokens)
        #The Counter class from Python's collections module creates a dictionary where:
        #Keys are unique words from the tokens list.
        #Values are the frequency (count) of each word in the list.
        features.append((Counter(tokens), label))

    return sorted(vocab), features


# Step 3: Train Naive Bayes
def train_naive_bayes(features, vocab):
    """
    Train a Naive Bayes model using the features and vocabulary.
    """
    class_totals = defaultdict(int)
    word_totals = defaultdict(lambda: defaultdict(int))
    class_counts = defaultdict(int)

    for word_counts, label in features:
        class_counts[label] += 1
        for word in vocab:
            word_totals[label][word] += word_counts[word]
            class_totals[label] += word_counts[word]

    # class_probs = P(Class) = Number of emails in the class/Total number of emails
    class_probs = {
        cls: log(count / sum(class_counts.values())) for cls, count in class_counts.items()
    }

    word_probs = {
        cls: {
            word: log((word_totals[cls][word] + 1) / (class_totals[cls] + len(vocab)))
            for word in vocab
        }
        for cls in class_counts
    }

    return class_probs, word_probs


# Step 4: Predict
def predict(text, class_probs, word_probs, vocab):
    """
    Predict the class of a given text using the trained Naive Bayes model.
    """
    tokens = preprocess(text)
    scores = {cls: class_probs[cls] for cls in class_probs}

    for cls in class_probs:
        for word in tokens:
            if word in vocab:
                scores[cls] += word_probs[cls][word]

    return max(scores, key=scores.get)


def write_info(output_file):
    """
    Write project information to a file.
    """
    info = {
        "Project_name": "SSOS",
        "Student_name": "Buica Ioana-Alexandra",
        "Alias_Student": "IO",
        "Project_version": "1",
    }

    try:
        with open(output_file, "w") as f:
            json.dump(info, f, indent=4)
        print(f"Information written to {output_file}.")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")


def scan_folder(folder, output_file):
    """
    Scan a folder for email files and classify them as spam or ham.
    """
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return

    # Train the spam detector model once
    vocab, features = create_vocab_and_features(training_data)
    class_probs, word_probs = train_naive_bayes(features, vocab)

    try:
        with open(output_file, "w") as file:
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)

                if os.path.isfile(item_path):
                    try:
                        with open(item_path, "r") as f:
                            data = f.readlines()

                        if data:
                            subject = data[0].strip()  # First line as subject
                            content = "".join(data[1:]).strip()  # Remaining lines as content

                            # Predict spam/ham
                            prediction = predict(f"{subject} {content}", class_probs, word_probs, vocab)

                            # Write results to output file
                            category = "inf" if prediction == "spam" else "cln"
                            file.write(f"{item}|{category}\n")
                            print(f"File: {item}, Prediction: {category}")

                    except Exception as e:
                        print(f"Error processing file '{item}': {e}")

        print(f"Classification results written to {output_file}.")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Anti-spam filter.")
    parser.add_argument(
        "-info",
        metavar="output_file",
        type=str,
        help="Write project info to the output file.",
    )
    parser.add_argument(
        "-scan",
        nargs=2,
        metavar=("folder", "output_file"),
        help="Scan the folder and classify files as spam or ham.",
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
