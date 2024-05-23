import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt


def save_confusion_matrix(y_true, y_pred, class_name, conf_file):
    """
    Create a confusion matrix with IndexName, Precision, Recall, F-Score, OA and Kappa
    Charlotte's style
    INPUT:
        - C: confusion_matrix compute by sklearn.metrics.confusion_matrix
        - class_name: corresponding name class
    OUTPUT:
        - conf_mat: Charlotte's confusion matrix
    """
    C = confusion_matrix(y_true, y_pred)

    nclass, _ = C.shape

    # -- Compute the different statistics
    recall = np.zeros(nclass)
    precision = np.zeros(nclass)
    fscore = np.zeros(nclass)
    diag_sum = 0
    hdiag_sum = 0
    for add in range(nclass):
        hdiag_sum = hdiag_sum + np.sum(C[add, :]) * np.sum(C[:, add])
        if C[add, add] == 0:
            recall[add] = 0
            precision[add] = 0
            fscore[add] = 0
        else:
            recall[add] = C[add, add] / np.sum(C[add, :])
            recall[add] = "%.6f" % recall[add]
            precision[add] = C[add, add] / np.sum(C[:, add])
            precision[add] = "%.6f" % precision[add]
            fscore[add] = (2 * precision[add] * recall[add]) / (
                precision[add] + recall[add]
            )
            fscore[add] = "%.6f" % fscore[add]
    nbSamples = np.sum(C)
    OA = np.trace(C) / nbSamples
    ph = hdiag_sum / (nbSamples * nbSamples)
    kappa = (OA - ph) / (1.0 - ph)

    #################### PANDA DATAFRAME #####################

    writer = pd.ExcelWriter(conf_file)

    line = [" "]
    for name in class_name:
        line.append(name)
    line.append("Recall")
    line = pd.DataFrame(np.array(line).reshape((1, -1)))
    line.to_excel(writer, startrow=0, header=False, index=False)

    row_ = 1
    for j in range(nclass):
        line = [class_name[j]]
        for i in range(nclass):
            line.append(str(C[j, i]))
        line.append(str(recall[j]))  # + '\n'
        line = pd.DataFrame(np.array(line).reshape((1, -1)))
        line.to_excel(writer, startrow=row_, header=False, index=False)
        row_ += 1

    line = ["Precision"]
    for add in range(nclass):
        line.append(str(precision[add]))
    line.append(str(OA))
    line.append(str(kappa))  # + '\n'
    line = pd.DataFrame(np.array(line).reshape((1, -1)))
    line.to_excel(writer, startrow=row_, header=False, index=False)
    row_ += 1

    line = ["F-Score"]
    for add in range(nclass):
        line.append(str(fscore[add]))
    line = pd.DataFrame(np.array(line).reshape((1, -1)))
    line.to_excel(writer, startrow=row_, header=False, index=False)

    writer.close()


def cross_scoring(filenames: list[str], classes, times, output_filename):

    OA_pix = np.zeros((1, times), dtype=float)
    Accuracy_classe_pix = np.zeros((len(classes), times), dtype=float)
    Rapel_classe_pix = np.zeros((len(classes), times), dtype=float)

    idx = 0
    for file in filenames:
        file_pix = pd.read_excel(file)

        OA_pix[0, idx] = file_pix.values[len(classes)][len(classes) + 1]

        for classe_idx in range(len(classes)):
            Accuracy_classe_pix[classe_idx, idx] = file_pix.values[len(classes)][
                classe_idx + 1
            ]
            Rapel_classe_pix[classe_idx, idx] = file_pix.values[classe_idx][
                len(classes) + 1
            ]

        idx += 1

    print(OA_pix)

    print(Accuracy_classe_pix)

    """
        Accuracy_classe_pix : la precision de chaque classe
        Rapel_classe_pix    : le rapel de chaque classe
    """

    writer = pd.ExcelWriter(output_filename)

    OA_pix_df = pd.DataFrame(np.array(OA_pix).reshape((1, -1)), index=["OA"])
    OA_pix_df.to_excel(writer, startrow=0, startcol=0)

    ####### Precision #######
    OA_pix_df = pd.DataFrame(["Precision"])
    OA_pix_df.to_excel(writer, startrow=4, startcol=0, index=False, header=False)

    OA_pix_df = pd.DataFrame(Accuracy_classe_pix, index=classes)
    OA_pix_df.to_excel(writer, startrow=5, startcol=0)

    ######## MAX ######
    OA_pix_df = pd.DataFrame([np.max(OA_pix)])
    OA_pix_df.to_excel(
        writer, startrow=0, startcol=times + 2, header=["MAX"], index=False
    )

    OA_pix_df = pd.DataFrame(np.max(Accuracy_classe_pix, axis=1))
    OA_pix_df.to_excel(
        writer, startrow=6, startcol=times + 2, index=False, header=False
    )

    ######## MIN ######
    OA_pix_df = pd.DataFrame([np.min(OA_pix)])
    OA_pix_df.to_excel(
        writer, startrow=0, startcol=times + 3, header=["MIN"], index=False
    )

    OA_pix_df = pd.DataFrame(np.min(Accuracy_classe_pix, axis=1))
    OA_pix_df.to_excel(
        writer, startrow=6, startcol=times + 3, index=False, header=False
    )

    ######## Medium ######
    OA_pix_df = pd.DataFrame([np.median(OA_pix)])
    OA_pix_df.to_excel(
        writer, startrow=0, startcol=times + 4, header=["Median"], index=False
    )

    OA_pix_df = pd.DataFrame(np.median(Accuracy_classe_pix, axis=1))
    OA_pix_df.to_excel(
        writer, startrow=6, startcol=times + 4, index=False, header=False
    )

    ######## AVG ######
    OA_pix_df = pd.DataFrame([np.average(OA_pix)])
    OA_pix_df.to_excel(
        writer, startrow=0, startcol=times + 5, header=["AVG"], index=False
    )

    OA_pix_df = pd.DataFrame(np.average(Accuracy_classe_pix, axis=1))
    OA_pix_df.to_excel(
        writer, startrow=6, startcol=times + 5, index=False, header=False
    )

    ######## STD ######
    OA_pix_df = pd.DataFrame([np.std(OA_pix)])
    OA_pix_df.to_excel(
        writer, startrow=0, startcol=times + 6, header=["STD"], index=False
    )

    OA_pix_df = pd.DataFrame(np.std(Accuracy_classe_pix, axis=1))
    OA_pix_df.to_excel(
        writer, startrow=6, startcol=times + 6, index=False, header=False
    )

    ####################################### Rapel #########################################"
    idx_line_rapel = len(classes) + 8
    OA_pix_df = pd.DataFrame(["Rapel"])
    OA_pix_df.to_excel(
        writer, startrow=idx_line_rapel, startcol=0, index=False, header=False
    )

    OA_pix_df = pd.DataFrame(Rapel_classe_pix, index=classes)
    OA_pix_df.to_excel(writer, startrow=idx_line_rapel + 1, startcol=0)

    ######## MAX ######
    OA_pix_df = pd.DataFrame(np.max(Rapel_classe_pix, axis=1))
    OA_pix_df.to_excel(
        writer,
        startrow=idx_line_rapel + 2,
        startcol=times + 2,
        index=False,
        header=False,
    )

    ######## MIN ######
    OA_pix_df = pd.DataFrame(np.min(Rapel_classe_pix, axis=1))
    OA_pix_df.to_excel(
        writer,
        startrow=idx_line_rapel + 2,
        startcol=times + 3,
        index=False,
        header=False,
    )

    ######## Medium ######
    OA_pix_df = pd.DataFrame(np.median(Rapel_classe_pix, axis=1))
    OA_pix_df.to_excel(
        writer,
        startrow=idx_line_rapel + 2,
        startcol=times + 4,
        index=False,
        header=False,
    )

    ######## AVG ######
    OA_pix_df = pd.DataFrame(np.average(Rapel_classe_pix, axis=1))
    OA_pix_df.to_excel(
        writer,
        startrow=idx_line_rapel + 2,
        startcol=times + 5,
        index=False,
        header=False,
    )

    ######## STD ######
    OA_pix_df = pd.DataFrame(np.std(Rapel_classe_pix, axis=1))
    OA_pix_df.to_excel(
        writer,
        startrow=idx_line_rapel + 2,
        startcol=times + 6,
        index=False,
        header=False,
    )

    writer.close()


def plot_confusion_matrix(
    cm: np.ndarray, classes: list[str], output_image: str, cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    """
    title = "Matrice de Confusion"

    # Compute confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = range(cm.shape[0])

    print("Confusion matrix")

    print(cm)
    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        # title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],  # format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()

    fig.savefig(output_image)


def generate_color_image(image_classes, lut):

    rows, cols = image_classes.shape
    colored_image = np.zeros((rows, cols, 3), dtype=np.uint8)

    for idx, item in lut["level2"].items():
        colored_image[image_classes == int(item["index"])] = item["color"]

    return colored_image
