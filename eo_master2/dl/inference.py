import os
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm


from eo_master2.dl.data_utils import ToTensor, Norm_percentile
from eo_master2.dl.dataloader import SITS
from eo_master2.dl.tempcnn import TempCNN
from eo_master2.evaluation import generate_color_image, save_confusion_matrix
from eo_master2.ml.data_utils import load_lut
from eo_master2 import tools


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lut_filename = "constants/level2_classes_labels.json"
    dirpath = (
        "/media/mohamed/Data/TeleDetection/DATA/interpolated/T31TCJ_Toulouse_Nord/"
    )
    classes = "/media/mohamed/Data/TeleDetection/DATA/T31TCJ_Toulouse_Nord_Classe.tif"
    # classes = "/media/mohamed/Data/TeleDetection/DATA/T31TCJ_Toulouse_Sud_Classe.tif"

    output_image = f"results/tempcnn_classification.png"
    output_xlsx = f"results/tempcnn_classification.xlsx"

    fold = 1
    model_filename = f"results/split_{fold}/tempcnn.pt"
    batch_size = 128

    lut = load_lut(lut_filename)
    class_labels = [i["name"] for i in lut.values()]

    # TempCNN parameters
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
    temp_cnn.load(model_filename)
    temp_cnn.to(device)
    temp_cnn.eval()

    transform = Compose(
        [
            ToTensor(),
        ]
    )
    test_set = SITS(
        dirpath=dirpath,
        lut_filename=lut_filename,
        classes_img_filename=classes,
        transform=transform,
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print("infering..")
    groud_truth = []
    predictions = []
    for batch in tqdm(test_loader, total=len(test_loader)):
        time_series, labels = batch[0].to(torch.float).to(device), batch[1].to(device)

        predicted = temp_cnn(time_series)
        _, predicted = torch.max(predicted.data, 1)

        groud_truth.extend(labels.cpu().numpy())
        predictions.extend(predicted.detach().cpu().numpy())

    save_confusion_matrix(groud_truth, predictions, class_labels, output_xlsx)

    
    predictions = np.array(predictions)
    predictions = predictions.reshape(test_set.get_shape())

    del test_loader, test_set

    colored_image = generate_color_image(predictions, lut)

    geoTransform, projection = tools.getGeoInformation(classes)
    tools.saveImage(
        colored_image, output_image, geoTransform=geoTransform, projection=projection
    )
