from collections import Counter
import os
from itertools import product
from typing import Optional, List
from torchvision.transforms import Compose
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from eo_master2.tools import readFiles, readImage, getGeoInformation
from eo_master2.ml.data_utils import load_data, load_lut
from eo_master2.dl.data_utils import ToTensor, Norm_percentile


class TemporalPixs(Dataset):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        nb_date: int = 182,
        transform: Optional[Compose] = None,
        level: str = "level1",
    ) -> None:

        self.X_train = X_train.reshape((-1, 4, nb_date))
        self.y_train = y_train

        self.transform = transform

    def __len__(self) -> int:
        return self.X_train.shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:

        time_series = self.X_train[index]
        label = self.y_train[index]

        if not self.transform is None:
            time_series = self.transform(time_series)

        return time_series, int(label)

    def get_class_weights(self):
        class_count = Counter(self.y_train)
        class_count = dict(sorted(class_count.items()))
        print(class_count)
        class_count = np.array([v for (t, v) in class_count.items()])
        class_count = class_count / class_count.sum()
        class_count = 1.0 / class_count
        print("Normalized weights des class : ", class_count)
        return class_count


class SITS(Dataset):
    """
    Satellite Image Time Series (SITS) Dataset.
    """

    def __init__(
        self,
        dirpath: str,
        lut_filename: str,
        classes_img_filename: str = None,
        sits_extention: str = ".tif",
        transform: Optional[Compose] = None,
    ) -> None:
        self.dirpath = dirpath

        # List of SITS image files
        self.sits_image_files = readFiles(self.dirpath, extension=sits_extention)
        self.sits_image_files = sorted(
            self.sits_image_files, key=lambda x: int(os.path.basename(x).split(".")[0])
        )

        # load lookUpTable (lut)
        self.lut = load_lut(lut_filename)

        # Read classes image if path is given
        self.classes = None
        if classes_img_filename:
            self.classes, _, _ = readImage(classes_img_filename)
            self.classes = self.classes[:, :, 0]

        self.transform = transform

        self._load_sits_images()

        self.rows, self.cols, _, _ = self.sits.shape
        self.coords = [(x, y) for x, y in product(range(self.rows), range(self.cols))]

    def _load_sits_images(self) -> np.ndarray:
        """
        Load all SITS images and transform to a 3D array.

        Returns:
            ndarray: 3D array of shape [index, Channels, time].
        """
        print("Loading SITS...")
        self.sits = []
        for image_file in tqdm(self.sits_image_files):
            img, _, _ = readImage(image_file)
            self.sits.append(img)

        self.sits = np.stack(self.sits, axis=0)  # [time, rows, cols, channels]
        self.sits = self.sits.transpose((1, 2, 3, 0))  # [rows, cols, channels, time]

    def __len__(self) -> int:
        """
        Get the number of images in the SITS dataset.

        Returns:
            int: Number of images in the SITS dataset.
        """
        return len(self.coords)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        x, y = self.coords[index]
        time_series = self.sits[x, y].astype(np.float64)

        # Apply transformation if specified
        if self.transform:
            time_series = self.transform(time_series)

        if not self.classes is None:
            classe = self.classes[x, y]
            return time_series, int(self.lut["level2"][str(classe)]["index"])
        else:
            return time_series

    def get_shape(self):
        return self.rows, self.cols

    def get_geodata(self):
        geoTransform, projection = getGeoInformation(self.sits_image_files[0])
        return geoTransform, projection


if __name__ == "__main__":
    lut_filename = "constants/level2_classes_labels.json"

    split_output_folder = "data/train_test_fold_1.npy"

    lut_filename = "constants/level2_classes_labels.json"

    lut = load_lut(lut_filename)

    X_train, y_train, _, _ = load_data(split_output_folder, lut)

    min_percentile = [157.0, 179.0, 207.0, 181.0]
    max_percentile = [3555.0, 4229.0, 3805.0, 2837.0]
    transform = Compose(
        [
            ToTensor(),
            Norm_percentile(
                np.array(min_percentile),
                np.array(max_percentile),
            ),
        ]
    )
    datset = TemporalPixs(X_train, y_train, transform=transform)

    dataloader = DataLoader(datset, batch_size=4, shuffle=True)

    for idx, (ts, l) in enumerate(dataloader):
        print(ts.max(), l)

        if idx == 10:
            break
