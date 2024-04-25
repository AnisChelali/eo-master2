import os 
import numpy as np
import pandas as pd
from collections import Counter
from . import tools

sits_folder = "G:/Memoire/data/interpolated/T31TCJ_Toulouse_Nord"
path_classes = "G:/Memoire/T31TCJ_Toulouse_Nord_Classe.tif"
path_conf = "G:/Memoire/T31TCJ_Toulouse_Nord_Confidence.tif"

list_images = tools.readFiles(sits_folder, ".tif")
list_images = sorted(list_images, key=lambda x: int(os.path.basename(x).split(".")[0]))

sits = []
for img in list_images:
    numpy_image, _, _ = tools.readImage(img)
    sits.append(numpy_image)
sits = np.stack(sits, axis=0) # => [Temps, Rows, COls, channels]

time, rows, cols, channels = sits.shape
sits = sits.transpose((1, 2, 3, 0))
sits = sits.reshape((-1, channels, time))


img_classes, _,_ = tools.readImage(path_classes)
img_confidence, _,_ = tools.readImage(path_conf)

img_classes = img_classes.reshape((-1, 1))
img_confidence = img_confidence.reshape((-1, 1))

classes = np.unique(img_classes)

nb_samples = 1000
cord = {}
for c in classes:
    x = np.argwhere(img_classes == c)
    indexs =np.argsort(img_confidence[x[: , 0]] , axis = 0).reshape((-1 , 1))
    a = x[indexs[: , 0][:: -1]][0 : nb_samples]
    b = img_confidence[a[: , 0]]
    print(c , "-->" , b.max())
    cord[c] = a  

timeSeries = []
labels = []

for c in classes:
    coord = cord[c] 
    pixel = sits[coord[: , 0]]
    print(pixel.shape)
    timeSeries.append(pixel)
    labels.extend([c]*len(coord))

timeSeries = np.vstack(timeSeries)
labels = np.array(labels)

np.save('../data/time_seriess', { 'labels' : labels , "timeSeries" : timeSeries })    

