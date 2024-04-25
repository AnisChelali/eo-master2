from itertools import product
import os
import time
import numpy as np
from tqdm import tqdm
from osgeo import gdal, ogr


from . import tools


def interpolate(x, xt, yt):
    """
    Perform linear interpolation to estimate y values at specified x positions.

    Parameters:
    - x (array-like): The x positions where interpolation is to be performed.
    - xt (array-like): The array of x positions of the known data points.
    - yt (array-like): The array of y values corresponding to the known data points.

    Returns:
    - yp (array): The interpolated y values at the specified x positions.

    This function removes data points with y values equal to zero, then performs linear interpolation
    to estimate y values at the given x positions. Linear interpolation calculates the y values for
    the specified x positions by interpolating between the adjacent known data points.

    Example:
    >>> xt = [1, 2, 4, 5]
    >>> yt = [3, 5, 0, 8]
    >>> x = [1.5, 3, 4.5]
    >>> interpolate(x, xt, yt)
    array([4., 1., 4.])
    """

    # Remove 0 values from the array
    selectes_positions = yt != 0
    x_filtred = xt[selectes_positions]
    y_filtred = yt[selectes_positions]

    # Perform linear interpolation
    yp = np.interp(x, x_filtred, y_filtred)

    return yp


def interpolate_sits(sits_folder: str, output_folder: str) -> None:
    """
    Perform temporal interpolation on Sentinel-2 Time Series (SITS) data and save the results.

    Parameters:
    - sits_folder (str): The path to the folder containing Sentinel-2 Time Series (SITS) data.
    - output_folder (str): The path to the folder where the interpolated SITS data will be saved.

    Returns:
    None

    This function performs temporal interpolation on Sentinel-2 Time Series (SITS) data using linear interpolation.
    The interpolated data is then stored in the specified output folder. The interpolation is performed over each
    band and temporal pixel of the SITS data.

    Example:
    >>> interpolate_sits("/path/to/sits_data", "/path/to/output_folder")
    """
    os.makedirs(output_folder, exist_ok=True)

    sits_files = tools.readFiles(sits_folder, ".tif")
    sits_files = sorted(
        sits_files, key=lambda x: int(os.path.basename(x).split(".")[0])
    )

    geotransform, projection = tools.getGeoInformation(sits_files[0])
    dates_2_days = np.array(tools.numJourAn_List(sits_files))

    sits = []
    for fimage in sits_files:
        image, _, _ = tools.readImage(fimage)
        sits.append(image)

    sits = np.stack(
        sits, axis=0
    )  # stack all images in a single array => [Nb date, Rows, Cols, Bands]

    new_days = np.arange(1, 365, 2)
    new_sits = np.zeros(
        (len(new_days), sits.shape[1], sits.shape[2], sits.shape[3]), dtype=sits.dtype
    )

    _, rows, cols, bands = sits.shape

    start_time = time.time()
    for d in range(bands):
        print("Process interpolation over band %d" % d)
        for x, y in tqdm(product(range(rows), range(cols)), total=rows * cols):
            new_sits[:, x, y, d] = interpolate(new_days, dates_2_days, sits[:, x, y, d])
    end_time = time.time()
    print("Interpolation process took %f seconds" % (end_time - start_time))

    del sits

    print("Storing interpolated sits...")
    start_time = time.time()
    for idx, date in enumerate(new_days):
        j, m, a = tools.datenumjouran(date, 2023)
        filename = "%04d%02d%02d" % (a, m, j)
        tools.saveImage(
            new_sits[idx],
            os.path.join(output_folder, "%s.tif" % filename),
            geotransform,
            projection,
            type=gdal.GDT_UInt16,
        )
    end_time = time.time()
    print("Storing process took %f seconds" % (end_time - start_time))

    del new_sits


if __name__ == "__main__":
    # crop = "T31TCJ_Toulouse_Nord"

    crops = [
        "T31TCJ_Toulouse_Nord",
        # "T31TCJ_Toulouse_Sud",
        # "T31SFA_BBA_BBA",
        # "T31SFA_BBA_Bejaya",
    ]
    for crop in crops:
        sits_folder = "E:/TeleDetection - Anis/DATA/raw/%s/" % crop
        output_folder = "E:/TeleDetection - Anis/DATA/interpolated/%s/" % crop
        output_folder_vizu = "E:/TeleDetection - Anis/DATA/interpolated/%s_vizu/" % crop
        interpolate_sits(sits_folder, output_folder)
        tools.sits_vizualization(output_folder, output_folder_vizu)
        # break
