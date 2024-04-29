import os
from typing import Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle as pkl

from eo_master2.ml.data_utils import *


# Function to train and evaluate the SVM classifier
def train_and_evaluate(input_file: str, output_folder):
    # Load the data
    data = np.load(input_file, allow_pickle=True).item()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

def load_data(filename: str) -> list[np.ndarray]:
    data = np.load(filename, allow_pickle=True).item()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    return X_train, y_train, X_test, y_test


# Function to train and evaluate the RF classifier
def train_knn_classifier(
    X_train: np.ndarray, y_train: np.ndarray, output_model_file: str = None
) -> RandomForestClassifier:
    # Initialize and train
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_classifier.fit(X_train, y_train)

    with open(output_folder, "wb") as file:
        pickle.dump(rf_classifier, file)

    return rf_classifier


def test_model(model: RandomForestClassifier, test_test: Union[list, np.ndarray]):
    # Predict and evaluate
    y_pred = model.predict(test_test)

    return y_pred


def evaluate(predicted_labels, groud_truth_labels, output_folder) -> None:

    accuracy = accuracy_score(predicted_labels, groud_truth_labels)

    return accuracy


accuracy_scores = []

# Loop over each fold and perform training and evaluation
for fold in range(1, 6):
    accuracy = train_and_evaluate(
        f"././data/train_test_fold_{fold}.npy", f"././results/split_{fold}/rf_model.pkl"
    )
    accuracy_scores.append(accuracy)

    input_data_folder = "data/"
    output_folder = "results/"

print(f"Mean Accuracy: {mean_accuracy}")
print(f"Standard Deviation: {std_accuracy}")
