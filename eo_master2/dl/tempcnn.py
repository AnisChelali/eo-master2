"""
Implementation of TempCNN with pytorch. 
This model is proposed by Pelletier et al. 2019 (https://www.mdpi.com/2072-4292/11/5/523)

Original repo: https://github.com/charlotte-pel/temporalCNN

This script is taken from: https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/models/TempCNN.py
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["TempCNN"]


class TempCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        sequence_length: int,
        kernel_size: int = 5,
        hidden_dims: int = 256,
        dropout: float = 0.5,
    ):
        """
        Temporal Convolutional Neural Network (TempCNN) for sequence classification.

        Args:
            input_dim (int): Number of input features.
            num_classes (int): Number of output classes.
            sequence_length (int): Length of the input sequence.
            kernel_size (int): Size of the convolutional kernels.
            hidden_dims (int): Number of hidden dimensions in the network.
            dropout (float): Dropout probability.

        Attributes:
            model_name (str): A string representing the model configuration.
            hidden_dims (int): Number of hidden dimensions.

        Modules:
            conv_bn_relu1 (Conv1D_BatchNorm_Relu_Dropout): First convolutional layer.
            conv_bn_relu2 (Conv1D_BatchNorm_Relu_Dropout): Second convolutional layer.
            conv_bn_relu3 (Conv1D_BatchNorm_Relu_Dropout): Third convolutional layer.
            flatten (Flatten): Flatten layer.
            dense (FC_BatchNorm_Relu_Dropout): Fully connected layer.
            log_softmax (nn.Sequential): LogSoftmax layer for output.

        Methods:
            forward(x): Forward pass through the network.
            save(path, **kwargs): Save the model to a specified path.
            load(path): Load the model from a specified path.
        """
        super(TempCNN, self).__init__()
        self.model_name = (
            f"TempCNN_input-dim={input_dim}_num-classes={num_classes}_sequence-length={sequence_length}_"
            f"kernel-size={kernel_size}_hidden-dims={hidden_dims}_dropout={dropout}"
        )

        self.hidden_dims = hidden_dims

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(
            input_dim, hidden_dims, kernel_size=kernel_size, drop_probability=dropout
        )
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(
            hidden_dims, hidden_dims, kernel_size=kernel_size, drop_probability=dropout
        )
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(
            hidden_dims, hidden_dims, kernel_size=kernel_size, drop_probability=dropout
        )
        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(
            hidden_dims * sequence_length, 4 * hidden_dims, drop_probability=dropout
        )
        self.log_softmax = nn.Sequential(
            nn.Linear(4 * hidden_dims, num_classes), nn.LogSoftmax(dim=-1)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the TempCNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Log probabilities of each class for each sequence in the batch.
        """

        # x = x.transpose(1, 2)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.log_softmax(x)

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to " + path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop("model_state", snapshot)
        self.load_state_dict(model_state)
        return snapshot


class Conv1D_BatchNorm_Relu_Dropout(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: int,
        kernel_size: int = 5,
        drop_probability: float = 0.2,
    ):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.conv1d = nn.Conv1d(
            input_dim,
            hidden_dims,
            kernel_size,
            padding="same",
            padding_mode="reflect",
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dims)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_probability)

        # Initialize the weights of the layers
        self._init_weights()

    def _init_weights(self):
        # Initialize weights of Conv1d layer using Xavier initialization
        nn.init.xavier_uniform_(self.conv1d.weight)
        nn.init.zeros_(self.conv1d.bias)  # Initialize bias to zeros

        # Initialize weights of BatchNorm1d layer
        nn.init.constant_(self.batch_norm.weight, 1)  # Set initial scale to 1
        nn.init.constant_(self.batch_norm.bias, 0)  # Set initial bias to 0

    def forward(self, x):
        # Perform convolution
        x = self.conv1d(x)

        # Apply batch normalization, ReLU, and dropout
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class FC_BatchNorm_Relu_Dropout(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: int, drop_probability: float = 0.2):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.linear = nn.Linear(input_dim, hidden_dims)
        self.batch_norm = nn.BatchNorm1d(hidden_dims)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_probability)

        # Initialize the weights of the layers
        self._init_weights()

    def _init_weights(self):
        # Initialize weights of Linear layer using Xavier initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)  # Initialize bias to zeros

        # Initialize weights of BatchNorm1d layer
        nn.init.constant_(self.batch_norm.weight, 1)  # Set initial scale to 1
        nn.init.constant_(self.batch_norm.bias, 0)  # Set initial bias to 0

    def forward(self, x):
        # Perform linear transformation
        x = self.linear(x)

        # Apply batch normalization, ReLU, and dropout
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class Flatten(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.view(input.size(0), -1)


if __name__ == "__main__":
    # Instantiate TempCNN
    sequence_length = 182  # time series length
    input_dim = 4
    kernel_size = 11
    hidden_dims = 128
    dropout = 0.3
    temp_cnn = TempCNN(
        input_dim=input_dim,
        kernel_size=kernel_size,
        hidden_dims=hidden_dims,
        num_classes=19,
        sequence_length=sequence_length,
    )
    # Set the model to evaluation mode
    temp_cnn.eval()

    # Generate random input data
    batch_size = 1
    input_data = torch.randn((batch_size, 4, 182))
    # Assuming input shape is (batch_size, input_dim, sequence_length)

    # Forward pass through TempCNN
    with torch.no_grad():
        output = temp_cnn(input_data)

    # Print the shape of the output
    print(f"Output shape: {output.shape}")

    model_pth = "checkpoints/TempCNN_11_128/split_1/TempCNN.pth"
    stat_dict = torch.load(model_pth, map_location=torch.device("cpu"))
    temp_cnn.load_state_dict(stat_dict)
    temp_cnn.eval()
