import os
import time
from typing import List, Union, Optional
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import matplotlib.pyplot as plt

from eo_master2.ml.data_utils import (
    load_data,
    load_lut,
    get_percentiles,
    check_dim_format,
)
from eo_master2.dl.data_utils import ToTensor, Norm_percentile
from eo_master2.dl.dataloader import TemporalPixs
from eo_master2.dl.tempcnn import TempCNN
from eo_master2.evaluation import save_confusion_matrix, cross_scoring


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


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lut_filename = "constants/level2_classes_labels.json"

    lut = load_lut(lut_filename)
    class_labels = [i["name"] for i in lut["level2"].values()]

    # TempCNN parameters
    sequence_length = 182  # time series length
    input_dim = 4
    kernel_size = 11
    hidden_dims = 256
    dropout = 0.3
    cross_csv_output = f"results/tempcnn_{kernel_size}_{hidden_dims}.xlsx"

    nb_folds = 5
    filenames = []
    for fold in range(1, nb_folds + 1):
        print("#################################")
        print(f"#              {fold}           #")
        split_output_folder_train = f"data/train_fold_{fold}.npy"
        split_output_folder_vald = f"data/vald_fold_{fold}.npy"
        split_output_folder_test = f"data/test_fold_{fold}.npy"
        model_output = f"results/split_{fold}/tempcnn_{kernel_size}_{hidden_dims}.pt"
        curve_output = (
            f"results/split_{fold}/tempcnn_{kernel_size}_{hidden_dims}_train.png"
        )
        csv_output = (
            f"results/split_{fold}/tempcnn_scores_{kernel_size}_{hidden_dims}.xlsx"
        )
        batch_size = 1024
        nb_epochs = 10000
        learning_rate = 1e-4
        early_stopping_patience = 10

        ###############################################""

        # Initialize EarlyStopping
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            verbose=True,
            path=model_output,
        )

        X_train, y_train = load_data(split_output_folder_train, lut)
        X_vald, y_vald = load_data(split_output_folder_vald, lut)

        # X = check_dim_format(X_train)
        # min_percentile, max_percentile = get_percentiles(X)
        # print(min_percentile)
        # print(max_percentile)

        # min_percentile = [157.0, 179.0, 207.0, 181.0]
        # max_percentile = [3555.0, 4229.0, 3805.0, 2837.0]
        transform = Compose(
            [
                ToTensor(),
                # Norm_percentile(
                #     np.array(min_percentile),
                #     np.array(max_percentile),
                # ),
            ]
        )
        print("Loading train set")
        train_set = TemporalPixs(X_train, y_train, transform=transform)
        print("Loading validationn set")
        validation_set = TemporalPixs(X_vald, y_vald, transform=transform)
        class_weights = train_set.get_class_weights()

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(
            validation_set, batch_size=batch_size, shuffle=False
        )

        temp_cnn = TempCNN(
            input_dim=input_dim,
            kernel_size=kernel_size,
            hidden_dims=hidden_dims,
            num_classes=len(class_weights),
            sequence_length=sequence_length,
        )
        temp_cnn.to(device)

        # definision des fonctions d'entrainnement
        criterion = torch.nn.CrossEntropyLoss(
            torch.FloatTensor(class_weights).to(device)
        )

        optimizer = Adam(
            temp_cnn.parameters(),
            lr=learning_rate,
            weight_decay=5.181869707846283e-05,
            betas=(0.9, 0.999),
        )

        print("Trainning...")
        t1 = time.time()
        trainning_losses = []
        validation_losses = []
        for i in range(nb_epochs):
            trainning_loss = 0.0
            temp_cnn.train()
            for batch in train_loader:
                time_series, labels = batch[0].to(torch.float).to(device), batch[1].to(
                    device
                )

                predicted: torch.Tensor = temp_cnn(time_series)

                loss = criterion(predicted, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                trainning_loss += loss.item()

            trainning_loss = trainning_loss / len(train_loader)
            trainning_losses.append(trainning_loss)

            validation_loss = 0.0
            temp_cnn.eval()
            for batch in validation_loader:
                time_series, labels = batch[0].to(device), batch[1].to(device)
                predicted = temp_cnn(time_series)
                loss = criterion(predicted, labels)
                validation_loss += loss.item()

            validation_loss = validation_loss / len(validation_loader)
            validation_losses.append(validation_loss)

            print(
                f"Epoch {i}; trainning loss: {trainning_loss}, validation loss: {validation_loss}"
            )
            if early_stopping(validation_loss, temp_cnn):
                print("Early stopping triggered.")
                break

        t2 = time.time()
        trainning_time = t2 - t1
        with open(f"{model_output}.txt", "wt") as fout:
            fout.write(str(trainning_time))

        plt.figure()
        plt.plot(trainning_losses, label="Train loss")
        plt.plot(validation_losses, label="Vald loss")
        plt.axvline(
            x=len(validation_losses) - early_stopping_patience - 1,
            color="red",
            label="Early stopping",
        )
        plt.title("Courbes de pertes")
        plt.xlabel("Epoch")
        plt.ylabel("Erreur")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(curve_output)
        del train_loader, validation_loader, X_train, y_train, X_vald, y_vald
        ##############################################################
        #                      Testing the model                     #
        print("Loading testset...")
        X_test, y_test = load_data(split_output_folder_test, lut)

        test_set = TemporalPixs(X_test, y_test, transform=transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        temp_cnn.load(model_output)
        temp_cnn.eval()

        print("infering...")
        groud_truth = []
        predictions = []
        for batch in test_loader:
            time_series, labels = batch[0].to(device), batch[1].to(device)

            predicted = temp_cnn(time_series)
            _, predicted = torch.max(predicted.data, 1)

            groud_truth.extend(labels.cpu().numpy())
            predictions.extend(predicted.detach().cpu().numpy())

        save_confusion_matrix(groud_truth, predictions, class_labels, csv_output)
        del test_loader, test_set, groud_truth, predictions, temp_cnn

        filenames.append(csv_output)

    cross_scoring(filenames, class_labels, nb_folds, cross_csv_output)
