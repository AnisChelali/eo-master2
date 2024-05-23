from sklearn.model_selection import StratifiedKFold
import numpy as np
import pickle as pkl


input_file = "data/time_series_100000.npy"
# Load the dataset
# dataset = np.load(input_file, allow_pickle=True)
with open(input_file, "rb") as f:
    dataset = pkl.load(f)

labels = dataset["labels"]
time_series = dataset["timeSeries"]

nb_data = len(labels)

# Flatten the time series data
X_flattened = time_series.reshape(time_series.shape[0], -1)

# Initialize StratifiedKFold with 5 splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


# Initialize lists to store train, validation, and test indices for each split
train_indices_list = []
valid_indices_list = []
test_indices_list = []

# Split the data and save to .npy files
for fold, (train_index, test_index) in enumerate(
    skf.split(X_flattened, labels), start=1
):
    # Further split train indices into train and validation indices
    train_valid_split = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    train_indices, valid_indices = next(
        train_valid_split.split(train_index, labels[train_index])
    )

    X_train, X_vald, X_test = (
        X_flattened[train_index[train_indices]],
        X_flattened[train_index[valid_indices]],
        X_flattened[test_index],
    )
    y_train, y_vald, y_test = (
        labels[train_index[train_indices]],
        labels[train_index[valid_indices]],
        labels[test_index],
    )

    print("X_train ", X_train.shape, len(y_train) / nb_data)
    print("X_vald ", X_vald.shape, len(y_vald) / nb_data)
    print("X_test ", X_test.shape, len(y_test) / nb_data)

    # Save the train and test splits for each fold
    with open(f"data/train_fold_{fold}.npy", "wb") as fout:
        pkl.dump({"X": X_train, "y": y_train}, fout, protocol=4)

    with open(f"data/vald_fold_{fold}.npy", "wb") as fout:
        pkl.dump({"X": X_vald, "y": y_vald}, fout, protocol=4)

    with open(f"data/test_fold_{fold}.npy", "wb") as fout:
        pkl.dump({"X": X_test, "y": y_test}, fout, protocol=4)
