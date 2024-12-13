import os
import re
import chardet
import numpy as np
from collections import Counter
from bs4 import BeautifulSoup

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']

# Helper function to read and preprocess email files (non-recursive)
def read_and_preprocess_non_recursive(folder):
    # List all files in the directory (non-recursively)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    all_words = []
    for file in files:
        encoding = detect_encoding(file)  # Detect file encoding
        try:
            with open(file, 'r', encoding=encoding, errors='ignore') as f:
                content = f.readlines()
        except UnicodeDecodeError:
            # If the detected encoding fails, try opening with a different encoding or using a fallback
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.readlines()
        
        if not content:
            continue
        
        # Extract subject (first line) and body (remaining lines)
        subject = content[0].strip()
        body = ''.join(content[1:]).strip()
        
        # Clean HTML content (if any)
        body = clean_html(body)
        
        # Remove short words (1-2 chars)
        body = re.sub(r'\b\w{1,2}\b', '', body)
        
        words = body.split()
        all_words.extend(words)
    return files, all_words

# Function to clean HTML content using BeautifulSoup
def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()

# Step 1: Build Dictionary (using the same approach as before)
# Paths to your folders
train_nonspam_folder = r"C:\Users\Io\Desktop\Lot1_\Lot1_\Lot1\Clean"
train_spam_folder = r"C:\Users\Io\Desktop\Lot1_\Lot1_\Lot1\Spam"

# Read and process non-spam and spam emails
_, all_words_nonspam = read_and_preprocess_non_recursive(train_nonspam_folder)
_, all_words_spam = read_and_preprocess_non_recursive(train_spam_folder)

# Combine words from both folders
all_words = all_words_nonspam + all_words_spam
word_counts = Counter(all_words)

# Select the top 2500 most frequent words
dictionary = [word for word, _ in word_counts.most_common(2500)]

# Save dictionary to file
with open('dictionary.txt', 'w', encoding='utf-8') as f:
    for idx, word in enumerate(dictionary, 1):
        f.write(f"{idx}. {word}\n")

# Helper function to build feature matrix and labels
def build_features_and_labels(folder, dictionary, label_value, max_docs=None):
    files, _ = read_and_preprocess_non_recursive(folder)
    features = []
    labels = []
    for doc_id, file in enumerate(files, 1):
        encoding = detect_encoding(file)
        try:
            with open(file, 'r', encoding=encoding, errors='ignore') as f:
                content = f.readlines()
        except UnicodeDecodeError:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.readlines()
        
        if not content:
            continue
        
        # Extract subject and body as before
        subject = content[0].strip()
        body = ''.join(content[1:]).strip()
        body = clean_html(body)
        
        word_counts = Counter(body.split())
        for word, count in word_counts.items():
            if word in dictionary:
                word_id = dictionary.index(word) + 1
                features.append([doc_id, word_id, count])
        labels.append(label_value)
        if max_docs and doc_id >= max_docs:
            break
    return features, labels

# Build training features and labels
features_nonspam, labels_nonspam = build_features_and_labels(train_nonspam_folder, dictionary, 0, max_docs=50)
features_spam, labels_spam = build_features_and_labels(train_spam_folder, dictionary, 1, max_docs=50)

train_features = np.array(features_nonspam + features_spam)
train_labels = np.array(labels_nonspam + labels_spam)

# Save to files
np.savetxt('train-features.txt', train_features, fmt='%d', delimiter=' ')
np.savetxt('train-labels.txt', train_labels, fmt='%d')

# Step 2: Build test features and labels (you can update the test folder paths similarly)
test_nonspam_folder = r"C:\Users\Io\Desktop\Lot1_\Lot1_\Lot1\Clean"
test_spam_folder = r"C:\Users\Io\Desktop\Lot1_\Lot1_\Lot1\Spam"

features_nonspam_test, labels_nonspam_test = build_features_and_labels(test_nonspam_folder, dictionary, 0)
features_spam_test, labels_spam_test = build_features_and_labels(test_spam_folder, dictionary, 1)

test_features = np.array(features_nonspam_test + features_spam_test)
test_labels = np.array(labels_nonspam_test + labels_spam_test)

# Save to files
np.savetxt('test-features.txt', test_features, fmt='%d', delimiter=' ')
np.savetxt('test-labels.txt', test_labels, fmt='%d')

# Step 3: Train Naive Bayes Classifier (same as before)
num_train_docs = 100
num_tokens = 2500

# Load training data
from scipy.sparse import coo_matrix

train_matrix = coo_matrix((train_features[:, 2],
                           (train_features[:, 0] - 1, train_features[:, 1] - 1)),
                          shape=(num_train_docs, num_tokens)).toarray()
train_labels = np.loadtxt('train-labels.txt', dtype=int)

# Calculate word probabilities
spam_indices = np.where(train_labels == 1)[0]
nonspam_indices = np.where(train_labels == 0)[0]

spam_wc = train_matrix[spam_indices].sum()
nonspam_wc = train_matrix[nonspam_indices].sum()

prob_token_spam = (train_matrix[spam_indices].sum(axis=0) + 1) / (spam_wc + num_tokens)
prob_token_nonspam = (train_matrix[nonspam_indices].sum(axis=0) + 1) / (nonspam_wc + num_tokens)

# Step 4: Test Naive Bayes Classifier (same as before)
test_matrix = coo_matrix((test_features[:, 2],
                          (test_features[:, 0] - 1, test_features[:, 1] - 1)),
                         shape=(len(test_labels), num_tokens)).toarray()

# Probability of spam (prior)
prob_spam = len(spam_indices) / num_train_docs

# Calculate log probabilities for each email
log_prob_spam = test_matrix @ np.log(prob_token_spam) + np.log(prob_spam)
log_prob_nonspam = test_matrix @ np.log(prob_token_nonspam) + np.log(1 - prob_spam)

# Make predictions
predictions = log_prob_spam > log_prob_nonspam

# Step 5: Evaluate Classifier
wrong_classifications = np.sum(predictions != test_labels)
error_rate = wrong_classifications / len(test_labels)

print(f"Error Rate: {error_rate:.4f}")
