import os
import re
from typing import Tuple
import numpy as np
from osgeo import gdal, osr


def numjouran(d):
    """Donne le numéro du jour dans l'année de la date d=[j,m,a] (1er janvier = 1, ...)"""
    j, m, a = d
    if (a % 4 == 0 and a % 100 != 0) or a % 400 == 0:  # bissextile?
        return (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366)[m - 1] + j
    else:
        return (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)[m - 1] + j


def numJourAn_List(files):
    """Donne le numero du jour dans l'année pour une liste de fichier donner
    le nom du fichier est de type : "20080225.*"
    """
    numJourList = list()
    for da in files:
        # print("=======", da.split("\\"))
        # x = len(da.split("/"))
        d = re.findall("[0-9]{2}", os.path.basename(da))
        # print(da, "-->", d, " | ", da.split(os.sep)[-1])
        d = [int(d[3]), int(d[2]), int(d[0] + d[1])]
        # print(d, "==>", numjouran(d))
        numJourList.append(numjouran(d))
    return numJourList


def datenumjouran(n, a):
    """Donne la date d=[j,m,a] qui est le nième jour de l'année a
    exemple :
       print datenumjouran(345,2008)  #  affiche [10, 12, 2008]
    """
    if (a % 4 == 0 and a % 100 != 0) or a % 400 == 0:  # bissextile?
        jm = (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366)
    else:
        jm = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)
    for m in range(1, 13):
        if jm[m] >= n:
            return [n - jm[m - 1], m, a]


def readFiles(path, extension):
    """
        Cette fonction permet d'identifier les fichier dans un dossier et de retourner ceux qui ont certaine extension
    :param path: chemin du dossier
    :return: files: liste des images .tif en chemin absolue
    """

    files = []
    for dirpath, dirname, filesname in os.walk(path):
        for filename in filesname:
            if filename.endswith(extension):
                # print(os.path.join(dirpath, filename))
                files.append(os.path.join(dirpath, filename))
    # print files
    return files


def toNDVI(a):
    """
    On change d'espace de couleur de [NIR,R,G,B,] vers NDVI qui ppermet de mieux localiser la vegetation
    """
    # ok = (a[:,:,0].astype(np.double)+a[:,:,1].astype(np.double)) != 0
    ndvi = (a[:, :, 0].astype(np.double) - a[:, :, 1].astype(np.double)) / (
        a[:, :, 0].astype(np.double) + a[:, :, 1].astype(np.double)
    )
    ndvi[np.isnan(ndvi)] = -1
    ndvi[ndvi == np.inf] = 1.0
    ndvi[ndvi == -float(np.inf)] = -1.0
    return ndvi


def toNDWI(a):
    """
    On change d'espace de couleur de [NIR,R,G,B,] vers NDWI qui ppermet de mieux localiser l'eaux
    """
    ndwi = (a[:, :, 2].astype(np.double) - a[:, :, 0].astype(np.double)) / (
        a[:, :, 2].astype(np.double) + a[:, :, 0].astype(np.double)
    )
    ndwi[np.isnan(ndwi)] = -1
    ndwi[ndwi == np.inf] = 1.0
    ndwi[ndwi == -float(np.inf)] = -1.0
    return ndwi


def toSAVI(a):
    """
    indice de végétation ajusté pour le sol.
    L est une constante elle fixé a 0.5
    """

    return (
        (a[:, :, 0].astype(np.double) - a[:, :, 1].astype(np.double))
        / (a[:, :, 0].astype(np.double) + a[:, :, 1].astype(np.double) + 0.5)
    ) * (1 + 0.5)


def toMSAVI(a):
    """
    L'indice modifié de végétation ajusté au sol (MSAVI2) tente de minimiser l'effet du sol nu sur l'indice de végétation ajusté au sol (SAVI).

    Référence : Qi, J. et al., 1994, "A modified soil vegetation adjusted index", Remote Sensing of Environment, Vol. 48, No. 2, 119–126.
    """

    msavi = a.astype(np.double)

    msavi = (1 / 2) * (
        2 * (a[:, :, 0] + 1)
        - np.sqrt(
            (2 * a[:, :, 0] + 1) * (2 * a[:, :, 0] + 1) - 8 * (a[:, :, 0] - a[:, :, 1])
        )
    )

    msavi[np.isnan(msavi)] = -1
    msavi[msavi == np.inf] = 1.0
    msavi[msavi == -float(np.inf)] = -1.0

    return msavi


def toIB(a):
    """
        Calcul de l'indice de briance
    :param a:
    :return:
    """
    return np.sqrt(
        a[:, :, 1].astype(np.double) ** 2 + a[:, :, 0].astype(np.double) ** 2
    )


def normalisation(
    mat: np.ndarray,
    mask: np.ndarray = None,
    percentile: float = 2,
) -> np.ndarray:
    """
        Ici on normalize les valeur entre 0 et 255 pour pouvoir visualiser les image et aussi les enregister


    Args:
        mat (np.ndarray): _description_
        mask (np.ndarray, optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    _, _, d = mat.shape
    if not mask is None:
        min_percentile = np.percentile(mat[mask != 0], q=percentile, axis=0)
        max_percentile = np.percentile(mat[mask != 0], q=100 - percentile, axis=0)
    else:
        min_percentile = np.percentile(mat, q=percentile, axis=(0, 1))
        max_percentile = np.percentile(mat, q=100 - percentile, axis=(0, 1))

    for i in range(d):
        mat[:, :, i][mat[:, :, i] > max_percentile[i]] = max_percentile[i]
        mat[:, :, i][mat[:, :, i] < min_percentile[i]] = min_percentile[i]

    return (mat - min_percentile) / (max_percentile - min_percentile)


def readImage(imagePath: str):
    """read geographic information from a reference image

    Args:
        imagePath (str): _description_

    Returns:
        _type_: _description_
    """
    raster_image = gdal.Open(imagePath)
    try:
        geoTransform = raster_image.GetGeoTransform()
        """
        geoTransform is a tuple of 6 elements:
            [0] left X coordinate
            [1] pixel width
            [2] row rotation (usually zero)
            [3] top Y coordinate
            [4] column rotation (usually zero)
            [5] pixel height, this will be negative for north up images
        """
        projection = raster_image.GetProjection()
    except:
        geoTransform = None
        projection = None

    number_bands = raster_image.RasterCount
    # print("number of bands ", number_bands)

    numpy_image = []
    for i in range(1, number_bands + 1):
        band = raster_image.GetRasterBand(i).ReadAsArray()
        numpy_image.append(band)

    numpy_image = np.dstack(numpy_image)

    return numpy_image, geoTransform, projection


def saveImage(
    img, path, geoTransform=None, projection=None, type=gdal.GDT_Byte, driver="GTiff"
):
    """for a given numpy.ndarray, save it in the disk by setting the geographic information or not

    Args:
        img (_type_): input image
        path (_type_): path of the output image
        geoTransform (_type_, optional): geographic transformation. Defaults to None.
        projection (_type_, optional): geographic projection. Defaults to None.
        type (_type_, optional): gdal image type. Defaults to gdal.GDT_Byte.

    """
    if len(img.shape) == 2:
        rows, cols = img.shape
        d = 1
    else:
        rows, cols, d = img.shape

    # print(rows, cols)
    if driver == "PNG":
        # create destination:
        dst_driver_tmp = gdal.GetDriverByName("MEM")
        outdata = dst_driver_tmp.Create("", xsize=cols, ysize=rows, bands=d, eType=type)
    else:
        gdriver = gdal.GetDriverByName(driver)
        outdata = gdriver.Create(path, cols, rows, d, type)  # gdal.GDT_Byte)

    if outdata is None:
        return ValueError(path)

    if not geoTransform is None:
        outdata.SetGeoTransform(geoTransform)  ##sets same geotransform as input
    if not projection is None:
        outdata.SetProjection(projection)  ##sets same projection as input

    if d == 1:
        outdata.GetRasterBand(1).WriteArray(img)
    else:
        for i in range(d):
            outdata.GetRasterBand(i + 1).WriteArray(img[:, :, i])

    if driver == "PNG":
        dst_driver = gdal.GetDriverByName("PNG")
        outdata_tmp = dst_driver.CreateCopy(path, outdata)
        outdata_tmp.FlushCache()  ##saves to disk!!
        outdata_tmp = None

    outdata.FlushCache()  ##saves to disk!!
    outdata = None


def getGeoInformation(imagePath: str):
    """read geographic information from a reference image

    Args:
        imagePath (str): _description_

    Returns:
        _type_: _description_
    """
    referenced_image = gdal.Open(imagePath)
    try:
        geoTransform = referenced_image.GetGeoTransform()
        projection = referenced_image.GetProjection()
    except:
        return None, None

    return geoTransform, projection


def getLatitudeLongitude(
    imagePath: str,
) -> Tuple[
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
    float,
]:
    """
    Retrieves the latitude and longitude of the corners of an image, along with the rotation angle.

    Args:
        imagePath (str): The file path of the image.

    Returns:
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], float]: A tuple containing:
            - The latitude and longitude of the top-left corner.
            - The latitude and longitude of the top-right corner.
            - The latitude and longitude of the bottom-left corner.
            - The latitude and longitude of the bottom-right corner.
            - The rotation angle of the image.
    """
    ds = gdal.Open(imagePath)
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    # create the new coordinate system (WGS 84)
    new_cs = osr.SpatialReference()
    new_cs.ImportFromEPSG(4326)

    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(old_cs, new_cs)

    # get the point to transform, pixel (0,0) in this case
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()

    # Calculate coordinates of top-left corner
    tl_latlong = transform.TransformPoint(gt[0], gt[3])

    # Calculate coordinates of top-right corner
    tr_latlong = transform.TransformPoint(gt[0] + width * gt[1], gt[3])

    # Calculate coordinates of bottom-left corner
    bl_latlong = transform.TransformPoint(gt[0], gt[3] + height * gt[5])

    # Calculate coordinates of bottom-right corner
    br_latlong = transform.TransformPoint(gt[0] + width * gt[1], gt[3] + height * gt[5])

    return tl_latlong, tr_latlong, bl_latlong, br_latlong
