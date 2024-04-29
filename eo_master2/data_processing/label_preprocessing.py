import numpy as np
import json

filename = "data/time_series.npy"
constant_file = "constants/classes_labels.json"
output_file = "constants/level2_classes_labels.json"

# load labels
data = np.load(filename, allow_pickle=True).item()
_, y_train = data["timeSeries"], data["labels"]

# load contans
with open(constant_file, "rt") as fin:
    constants = json.load(fin)

level2 = list(constants["Level2"].values())
labels = np.sort(list(set(y_train))).astype(int)


labelisation = {}
for idx, c in enumerate(labels):
    c = int(c)
    label = list(filter(lambda x: x["code"] == c, level2))[0]
    labelisation[c] = {}
    labelisation[c]["index"] = idx
    labelisation[c]["gt"] = label["code"]
    labelisation[c]["name"] = label["alias"]
    labelisation[c]["color"] = label["RGB color"]
    labelisation[c]["parent"] = label["parent"]

with open(output_file, "wt") as fout:
    json.dump(labelisation, fout, indent=2, sort_keys=True)
