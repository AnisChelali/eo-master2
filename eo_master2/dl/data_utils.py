import numpy as np
import torch
from typing import Union


class ToTensor(object):
    def __call__(self, time_serie: np.ndarray):
        # if time_serie.dtype == np.uint8 or time_serie.dtype == np.int8:
        #     print("uint8")
        #     time_serie = time_serie / 255.0

        # if time_serie.dtype == np.uint16 or time_serie.dtype == np.int16:
        #     print("uint16")
        time_serie = time_serie / float(2**16 - 1)

        return torch.from_numpy(time_serie)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Norm_percentile(object):
    def __init__(
        self,
        min_percentil: Union[list, np.ndarray],
        max_percentil: Union[list, np.ndarray],
    ):
        self.min_percentil = torch.Tensor(min_percentil)
        self.max_percentil = torch.Tensor(max_percentil)

    def __call__(self, time_series: torch.Tensor):
        if not isinstance(time_series, torch.Tensor):
            return TypeError("input is not a tensor")

        for i in range(self.max_percentil.shape[0]):
            time_series[i] = (time_series[i] - self.min_percentil[i]) / (
                self.max_percentil[i] - self.min_percentil[i]
            )
        return time_series
