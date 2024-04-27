import numpy as np
import json

filename = "data/time_series.npy"
constant_file = "constants/classes_labels.json"

# load labels
data = np.load(filename, allow_pickle=True).item()
_, y_train = data["timeSeries"], data["labels"]

# load contans
with open(constant_file, "rt") as fin:
    constants = json.load(fin)

level2 = constants["Level2"]
labels = np.sort(list(set(y_train)))
