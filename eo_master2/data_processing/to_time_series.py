import os
import numpy as np
import pickle as pkl

from eo_master2 import tools

# root = "G:/Memoire/data"
root = "/media/mohamed/Data/TeleDetection/DATA"

sits_folder = f"{root}/interpolated/T31TCJ_Toulouse_Nord"
path_classes = f"{root}/T31TCJ_Toulouse_Nord_Classe.tif"
path_conf = f"{root}/T31TCJ_Toulouse_Nord_Confidence.tif"

list_images = tools.readFiles(sits_folder, ".tif")
list_images = sorted(list_images, key=lambda x: int(os.path.basename(x).split(".")[0]))
print("list_images : ", len(list_images))

sits = []
for img in list_images:
    numpy_image, _, _ = tools.readImage(img)
    sits.append(numpy_image)
sits = np.stack(sits, axis=0)  # => [Temps, Rows, COls, channels]

time, rows, cols, channels = sits.shape
sits = sits.transpose((1, 2, 3, 0))
sits = sits.reshape((-1, channels, time))


img_classes, _, _ = tools.readImage(path_classes)
img_confidence, _, _ = tools.readImage(path_conf)

img_classes = img_classes.reshape((-1, 1))
img_confidence = img_confidence.reshape((-1, 1))

classes = np.unique(img_classes)

nb_samples = 1000000
cord = {}
for c in classes:
    x = np.argwhere(img_classes == c)
    indexs = np.argsort(img_confidence[x[:, 0]], axis=0).reshape((-1, 1))

    a = x[indexs[:, 0][::-1]][0:nb_samples]
    b = img_confidence[a[:, 0]]
    print(c, "-->", b.max(), b.min())
    cord[c] = a

timeSeries = []
labels = []

for c in classes:
    coord = cord[c]
    pixel = sits[coord[:, 0]]
    print(pixel.shape)
    timeSeries.append(pixel)
    labels.extend([c] * len(coord))

timeSeries = np.vstack(timeSeries)
labels = np.array(labels)

# np.save(
#     f"data/time_series_{nb_samples}.npy",
#     {"labels": labels, "timeSeries": timeSeries},
#     allow_pickle=True,
# )

with open(f"data/time_series_{nb_samples}.npy", "wb") as fout:
    pkl.dump({"labels": labels, "timeSeries": timeSeries}, fout, protocol=4)
