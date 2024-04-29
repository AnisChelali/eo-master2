from typing import Optional, List
from torchvision.transforms import Compose
import numpy as np
from torch.utils.data import Dataset, DataLoader

from eo_master2.ml.data_utils import load_data, load_lut
from eo_master2.dl.data_utils import ToTensor, Norm_percentile


class TemporalPixs(Dataset):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        nb_date: int = 182,
        transform: Optional[Compose] = None,
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
