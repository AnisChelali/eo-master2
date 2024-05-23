import numpy as np
import json
import pickle as pkl


def categorize(lut, labels, level=2):
    level1 = lut["level1"]
    level2 = lut["level2"]
    if level == 2:
        new_labels = np.array([level2[str(i)]["index"] for i in labels])
    elif level == 1:
        new_labels = np.array(
            [level1[str(level2[str(i)]["parent"])]["index"] for i in labels]
        )
        print(new_labels, new_labels.dtype)
        print(np.unique(new_labels))

    return new_labels


def load_lut(filename: str):
    with open(filename, "rt") as fin:
        lut = json.load(fin)
    return lut


def load_data(filename: str, lut: dict) -> list[np.ndarray]:
    # data = np.load(filename, allow_pickle=True).item()
    with open(filename, "rb") as f:
        data = pkl.load(f)
    X, y = data["X"], data["y"]

    y = categorize(lut, y)
    return (
        X.astype(np.float32),
        y.astype(np.float32),
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
