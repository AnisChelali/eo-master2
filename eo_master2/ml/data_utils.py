import numpy as np
import json


def categorize(lut, labels):
    new_labels = np.array([lut[str(i)]["index"] for i in labels])
    return new_labels


def load_lut(filename: str):
    with open(filename, "rt") as fin:
        lut = json.load(fin)
    return lut


def load_data(filename: str, lut: dict) -> list[np.ndarray]:
    data = np.load(filename, allow_pickle=True).item()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    y_train = categorize(lut, y_train)
    y_test = categorize(lut, y_test)
    return (
        X_train.astype(np.float32),
        y_train.astype(np.float32),
        X_test.astype(np.float32),
        y_test.astype(np.float32),
    )


def get_percentiles(data_array: np.ndarray):
    min_per = np.percentile(data_array, q=2, axis=(0, 2))
    max_per = np.percentile(data_array, q=98, axis=(0, 2))
    return min_per, max_per


def check_dim_format(data_array, nb_date=182):
    if len(data_array.shape) < 3:
        data_array = data_array.reshape((-1, 4, nb_date))
    return data_array


def normelize(data_array, min_per, max_per):
    for i in range(data_array.shape[1]):
        data_array[:, i, :] = (data_array[:, i, :] - min_per[i]) / (
            max_per[i] - min_per[i]
        )
    return data_array
