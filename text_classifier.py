"""
Module Name: text_classifier.py
Description: A supervised method text binary classifier with different models.
Author: M. Mehdi Balouchi
Date: 26.10.2023
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import numpy as np
import string
import time


import numpy as np


download("stopwords")
ENGLISH_STOP_WORDS = list(set(stopwords.words("english")))


def read_data(path: str) -> list[tuple]:
    """
    Read and parse the tsv file a.

    Args:
        path (string): The path of the tsv file.
        ...

    Returns:
        list[tuple(doc, label)]: Returns the parsed data.
    """

    results = []

    with open(path, "r") as doc:  # use with to ensure file is closed properly
        next(doc)  # Ignore header
        for line in doc:
            text, label = line.split("\n")[0].split("\t")
            results.append((text, label))
    return results


def calculate_features(
    docs: list[tuple], tokenized_docs=None, vocabulary=None
) -> list[list]:
    """
    Calculate the features of the docs, using tokenized_docs and vocabulary for TF-IDF method.
    the current features are:
    1. Number of sentences in the doc.
    2. Mean number of words in each sentence.
    3. Mean number of characters in each word.
    4. TF-IDF of each word in the vocabulary.

    Args:
        docs (list): list of tuples of doc and label.
        tokenized_docs (list): list of tokenized documents.
        vocabulary (list): list of words in the vocabulary.

    Returns:
        list[features]: Returns the merged features for each doc.
    """

    features = []
    _tfs, _idfs, tf_idfs = TF_IDF(tokenized_docs, vocabulary)
    i = 0
    for doc, _ in docs:
        sentences = sent_tokenize(doc)
        num_sent = len(sentences)
        mean_words = np.mean([len(word_tokenize(sent)) for sent in sentences])
        mean_chars = np.mean([len(word) for word in word_tokenize(doc)])
        features.append([num_sent, mean_words, mean_chars] + list(tf_idfs[i]))

        i += 1
    return features


def remove_punctuation(tokens: list) -> list:
    """
    Remove the punctuations from the tokens.

    Args:
        tokens (list): List of the punctuations.

    Returns:
        list: Returns the tokens without punctuations.
    """

    return [word for word in tokens if word not in string.punctuation]


def remove_stop_words(tokens):
    """
    Removing the stop words from the tokens. based on nltk stopwords list.

    Args:
        tokens (list): list of tokens.

    Returns:
        list: List of tokens without stop words.
    """
    return [word for word in tokens if word not in ENGLISH_STOP_WORDS]


def extract_vocabulary(tokenized_docs: list, sort=True):
    """
    extract the vocabulary from the tokenized docs which are the unique words in the docs.

    Args:
        tokenized_docs (list): list of tokens.
        sort (bool, optional): Sort the vocabulary. Defaults to True.

    Returns:
        list: List of unique words in the docs.
    """

    vocabulary = set()
    for tokenized_doc in tokenized_docs:
        for word in tokenized_doc:
            vocabulary.add(word)
    vocabulary = list(vocabulary)
    if sort:
        return sorted(vocabulary)
    return vocabulary


def tokenize_docs(docs):
    """
    tokenize the docs using nltk word_tokenize and remove the punctuations and stop words.

    Args:
        docs (list): List of raw documents.

    Returns:
        list: List of tokenized docs.
    """
    tokenized_docs = []
    for doc, _ in docs:
        tokenized_doc = word_tokenize(doc)
        tokenized_doc = [token.lower() for token in tokenized_doc]
        tokenized_doc = remove_punctuation(tokenized_doc)
        tokenized_doc = remove_stop_words(tokenized_doc)
        tokenized_docs.append(tokenized_doc)
    return tokenized_docs


def TF_IDF(tokenized_docs, vocabulary=None):
    """
    Calculate the TF-IDF of the tokenized docs.

    Args:
        tokenized_docs (list): list of tokens.
        vocabulary (list, optional): list of words in the vocabulary. Defaults to None.

    Returns:
        (list, list, list): List of Tf list, list of idf, list of tf-idf.
    """

    tf_idfs = []
    docs_length = len(tokenized_docs)
    counters = [Counter(doc) for doc in tokenized_docs]

    if vocabulary is None:
        vocabulary = extract_vocabulary(tokenized_docs)

    # Number of docs that contain each token in the vocabulary
    total_frequencies = {key: 0 for key in vocabulary}
    empty_frequencies = total_frequencies.copy()
    tfs = []

    for counter in counters:
        for word in counter.keys():
            total_frequencies[word] += 1
        doc_length = len(counter)
        local_frequncy = empty_frequencies.copy()
        local_frequncy.update(counter)
        local_frequncy = dict(sorted(local_frequncy.items()))
        tfs.append(list(np.divide(list(local_frequncy.values()), doc_length)))

    idfs = dict(sorted(total_frequencies.items()))
    idfs = np.log(np.divide(docs_length, list(total_frequencies.values())))
    tf_idfs = np.multiply(tfs, idfs)

    return tfs, idfs, tf_idfs


def main():
    # Read the data
    train_raw = read_data("train.tsv")
    test_raw = read_data("test.tsv")

    # Extract the labels and docs
    train_labels = [label for _, label in train_raw]
    test_labels = [label for _, label in test_raw]

    # Tokenize the docs and extract the vocabulary for train and test
    train_tokenized_docs = tokenize_docs(train_raw)
    train_vocabulary = extract_vocabulary(train_tokenized_docs)

    test_tokenized_docs = tokenize_docs(train_raw)
    test_vocabulary = extract_vocabulary(train_tokenized_docs)

    # Calculate the features for train and test

    train_features = calculate_features(
        train_raw, train_tokenized_docs, train_vocabulary
    )

    test_features = calculate_features(test_raw, test_tokenized_docs, test_vocabulary)

    # Train the models and predict the test labels
    logistic_model = LogisticRegression(max_iter=500)
    mlp_model = MLPClassifier()

    logistic_model.fit(train_features, train_labels)
    mlp_model.fit(train_features, train_labels)

    test_pred_logistic = logistic_model.predict(test_features)
    test_pred_mlp = mlp_model.predict(test_features)

    # Calculate the accuracy and classification report

    accuracy_logistic = accuracy_score(test_labels, test_pred_logistic)
    report_logistic = classification_report(test_labels, test_pred_logistic)

    accuracy_mlp = accuracy_score(test_labels, test_pred_mlp)
    report_mlp = classification_report(test_labels, test_pred_mlp)

    print(f"Accuracy Logistic: {accuracy_logistic:.2f}")
    print(report_logistic)
    print("=================================================")
    print(f"Accuracy MLP: {accuracy_mlp:.2f}")
    print(report_mlp)


if __name__ == "__main__":
    print("Start time: ", time.strftime("%H:%M:%S", time.localtime()))
    main()
    print("End time: ", time.strftime("%H:%M:%S", time.localtime()))
