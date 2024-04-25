from sklearn.model_selection import StratifiedKFold
import numpy as np

# Load the dataset
dataset = np.load('time_series.npy', allow_pickle=True).item()
labels = dataset["labels"]
time_series = dataset["timeSeries"]

# Flatten the time series data
X_flattened = time_series.reshape(time_series.shape[0], -1)

# Initialize StratifiedKFold with 5 splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Split the data and save to .npy files
for fold, (train_index, test_index) in enumerate(skf.split(X_flattened, labels), start=1):
    X_train, X_test = X_flattened[train_index], X_flattened[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Save the train and test splits for each fold
    np.save(f'train_fold_{fold}.npy', {'X_train': X_train, 'y_train': y_train})
    np.save(f'test_fold_{fold}.npy', {'X_test': X_test, 'y_test': y_test})
