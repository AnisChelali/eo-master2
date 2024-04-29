from typing import List, Union, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader


class EarlyStopping:

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0,
        mode: str = "min",
        verbose: bool = False,
        path: Optional[str] = None,
    ):
        """
        Early stopping to stop training when the monitored quantity has stopped improving.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): One of 'min' or 'max'. In 'min' mode, training will stop when the quantity monitored has stopped decreasing. In 'max' mode, it will stop when the quantity monitored has stopped increasing.
            verbose (bool): If True, prints a message for each improvement.
            path (Optional[str]): Path to save the best model. If None, the model won't be saved.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "min":
            self.monitor_op = np.less
            self.best_score = float("inf")
        elif mode == "max":
            self.monitor_op = np.greater
            self.best_score = float("-inf")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Call method to check if training should stop early.

        Args:
            val_loss (float): Current value of the monitored quantity on the validation set.
            model (Union[torch.nn.Module, None]): PyTorch model to potentially save.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif self.monitor_op(val_loss, self.best_score - self.delta):
            self.save_checkpoint(val_loss, model)
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"\t\033[93mEarlyStopping counter: {self.counter} out of {self.patience}\033[0m"
                )

            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        """
        Save the model checkpoint when validation loss decreases.

        Args:
            val_loss (float): Current value of the monitored quantity on the validation set.
            model (torch.nn.Module): PyTorch model to save.
        """
        if self.verbose:
            print(
                f"\t\033[92mValidation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...\033[0m"
            )

        if self.path is not None:
            # with torch.no_grad():
            # torch.save(model.state_dict(), self.path)
            model.save(self.path)
