from sklearn.model_selection import StratifiedKFold
import numpy as np

# Load the dataset
dataset = np.load("data/time_series.npy", allow_pickle=True).item()
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
    np.save(
        f"data/train_fold_{fold}.npy",
        {"X_train": X_train, "y_train": y_train},
    )
    np.save(
        f"data/vald_fold_{fold}.npy",
        {"X_vald": X_vald, "y_vald": y_vald},
    )
    np.save(
        f"data/test_fold_{fold}.npy",
        {"X_test": X_test, "y_test": y_test},
    )
