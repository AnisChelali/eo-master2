import os
from typing import Union
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle as pkl

from eo_master2.ml.data_utils import *
from eo_master2.evaluation import save_confusion_matrix, cross_scoring


# Function to train and evaluate the KNN classifier
def train_knn_classifier(
    X_train: np.ndarray, y_train: np.ndarray, output_model_file: str = None
) -> KNeighborsClassifier:
    # Initialize and train
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    if not output_model_file is None:
        print(f"Saving model to: {output_model_file}...")
        with open(output_model_file, "wb") as file:
            pkl.dump(knn, file)

    return knn


def test_model(model: KNeighborsClassifier, test_test: Union[list, np.ndarray]):
    # Predict and evaluate
    y_pred = model.predict(test_test)

    return y_pred


if __name__ == "__main__":

    input_data_folder = "data/"
    lut_filename = "constants/level2_classes_labels.json"
    output_folder = "results/"
    cross_csv_output = "results/KNeighborsClassifier.xlsx"

    lut = load_lut(lut_filename)
    class_labels = [i["name"] for i in lut.values()]

    filenames = []
    # Loop over each fold and perform training and evaluation
    for fold in range(1, 6):
        csv_output = f"results/split_{fold}/knn_scores.xlsx"

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

        model = train_knn_classifier(
            X_train=X_train,
            y_train=y_train,
            output_model_file=f"{split_output_folder}knn_model.pkl",
        )

        y_predicted = test_model(model=model, test_test=X_test)

        save_confusion_matrix(y_test, y_predicted, class_labels, csv_output)
        filenames.append(csv_output)

    cross_scoring(filenames, class_labels, 5, cross_csv_output)
