import os
from typing import Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle as pkl


def load_data(filename: str) -> list[np.ndarray]:
    data = np.load(filename, allow_pickle=True).item()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    return X_train, y_train, X_test, y_test


# Function to train and evaluate the rf classifier
def train_knn_classifier(
    X_train: np.ndarray, y_train: np.ndarray, output_model_file: str = None
) -> RandomForestClassifier:
    # Initialize and train
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_classifier.fit(X_train, y_train)

    if not output_model_file is None:
        print(f"Saving model to: {output_model_file}...")
        with open(output_model_file, "wb") as file:
            pkl.dump(rf_classifier, file)

    return rf_classifier


def test_model(model: RandomForestClassifier, test_test: Union[list, np.ndarray]):
    # Predict and evaluate
    y_pred = model.predict(test_test)

    return y_pred


def evaluate(predicted_labels, groud_truth_labels, output_folder) -> None:

    accuracy = accuracy_score(predicted_labels, groud_truth_labels)

    return accuracy


if __name__ == "__main__":

    input_data_folder = "data/"
    output_folder = "results/"

    accuracy_scores = []
    # Loop over each fold and perform training and evaluation
    for fold in range(1, 6):

        split_output_folder = f"{output_folder}split_{fold}/"
        os.makedirs(split_output_folder, exist_ok=True)

        X_train, y_train, X_test, y_test = load_data(
            filename=f"{input_data_folder}train_test_fold_{fold}.npy"
        )

        model = train_knn_classifier(
            X_train=X_train,
            y_train=y_train,
            output_model_file=f"{split_output_folder}rf_model.pkl",
        )

        y_predicted = test_model(model=model, test_test=X_test)
        accuracy = evaluate(
            predicted_labels=y_predicted,
            groud_truth_labels=y_test,
            output_folder=split_output_folder,
        )

        accuracy_scores.append(accuracy)

    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)

    print(f"Mean Accuracy: {mean_accuracy}")
    print(f"Standard Deviation: {std_accuracy}")
