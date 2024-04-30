import os
from typing import Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle as pkl

from eo_master2.ml.data_utils import *


# Function to train and evaluate the RF classifier
def train_rf_classifier(
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
    lut_filename = "constants/level2_classes_labels.json"
    output_folder = "results/"

    lut = load_lut(lut_filename)

    accuracy_scores = []
    # Loop over each fold and perform training and evaluation
    for fold in range(1, 6):

        split_output_folder = f"{output_folder}split_{fold}/"
        os.makedirs(split_output_folder, exist_ok=True)

        X_train, y_train = load_data(
            filename=f"{input_data_folder}train_fold_{fold}.npy", lut=lut
        )
        X_test, y_test = load_data(
            filename=f"{input_data_folder}test_fold_{fold}.npy", lut=lut
        )

        # Data preparation
        X_train = check_dim_format(X_train, nb_date=182)
        X_test = check_dim_format(X_test, nb_date=182)

        min_per, max_per = get_percentiles(X_train)

        X_train = normelize(X_train, min_per, max_per)
        X_test = normelize(X_test, min_per, max_per)

        X_train = X_train.reshape((-1, 182 * 4))
        X_test = X_test.reshape((-1, 182 * 4))

        model = train_rf_classifier(
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

