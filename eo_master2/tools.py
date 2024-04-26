import os
import re
import numpy as np
from osgeo import gdal, ogr


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


def sits_vizualization(input_folder: str, output_foler: str) -> None:
    os.makedirs(output_foler, exist_ok=True)

    list_images = readFiles(input_folder, ".tif")
    for idx, fimage in enumerate(list_images):
        # print(idx, " -> ", fimage)
        numpy_image, geoTransform, projection = readImage(fimage)

        numpy_image = (
            normalisation(numpy_image[:, :, 0:3], percentile=2) * 255
        ).astype(np.uint8)

        saveImage(
            numpy_image,
            os.path.join(
                output_foler, "%s.png" % os.path.basename(fimage).split("/")[0]
            ),
            geoTransform=geoTransform,
            projection=projection,
            type=gdal.GDT_Byte,
            driver="PNG",
        )


def crop_geoimage(
    image_filepath: str, x1: int, y1: int, x2: int, y2: int, output_folder: str
) -> None:
    """_summary_

    Args:
        image_filepath (str): chemin de l'image à cropper
        x1 (int): point de depart
        y1 (int): point de depart
        x2 (int): point d'arriver
        y2 (int): point d'arriver
        output_folder (str): Chemin ou enregistrer les images croppé
    """
    image = gdal.Open(image_filepath)

    ########### Calcul du nouveau GeoReferencement #############
    # recuperer l'ancien georeferencement
    transform = image.GetGeoTransform()
    projection = image.GetProjection()

    new_transformation = (
        transform[0] + y1 * transform[1],
        transform[1],
        transform[2],
        transform[3] + x1 * transform[5],
        transform[4],
        transform[5],
    )
    # print ("Olt transformation \t", transform)
    # print ("New Transformation \t", new_transformation)

    ############################################################
    ############################################################
    nir_bands = image.GetRasterBand(1).ReadAsArray()
    nir_bands = nir_bands[x1:x2, y1:y2]
    r_bands = image.GetRasterBand(2).ReadAsArray()
    r_bands = r_bands[x1:x2, y1:y2]
    g_bands = image.GetRasterBand(3).ReadAsArray()
    g_bands = g_bands[x1:x2, y1:y2]
    b_bands = image.GetRasterBand(4).ReadAsArray()
    b_bands = b_bands[x1:x2, y1:y2]

    # print( type(b_bands[0,0]))
    # print(b_bands.shape)
    [cols, rows] = b_bands.shape

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(
        os.path.join(output_folder, os.path.basename(image_filepath)),
        rows,
        cols,
        4,
        gdal.GDT_UInt16,
    )
    # print(outdata)
    outdata.GetRasterBand(1).WriteArray(nir_bands)
    outdata.GetRasterBand(2).WriteArray(r_bands)
    outdata.GetRasterBand(3).WriteArray(g_bands)
    outdata.GetRasterBand(4).WriteArray(b_bands)

    outdata.SetProjection(projection)  ##sets same projection as input
    outdata.SetGeoTransform(new_transformation)  ##sets same geotransform as input

    outdata.FlushCache()  ##saves to disk!!
    outdata = None


def crop_gt_geoimage(
    image_filepath: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    output_image: str,
    type: int = gdal.GDT_Byte,
) -> None:
    """_summary_

    Args:
        image_filepath (str): chemin de l'image à cropper
        x1 (int): point de depart
        y1 (int): point de depart
        x2 (int): point d'arriver
        y2 (int): point d'arriver
        output_folder (str): Chemin ou enregistrer les images croppé
    """
    image = gdal.Open(image_filepath)

    ########### Calcul du nouveau GeoReferencement #############
    # recuperer l'ancien georeferencement
    transform = image.GetGeoTransform()
    projection = image.GetProjection()

    new_transformation = (
        transform[0] + y1 * transform[1],
        transform[1],
        transform[2],
        transform[3] + x1 * transform[5],
        transform[4],
        transform[5],
    )
    print("Olt transformation \t", transform)
    print("New Transformation \t", new_transformation)

    ############################################################
    ############################################################
    band = image.GetRasterBand(1).ReadAsArray()
    print("===+> ", band.shape)
    band = band[x1:x2, y1:y2]

    # print( type(b_bands[0,0]))
    print(band.shape)
    # print(gdal.GetDataTypeName(image.GetRasterBand(1).DataType))
    # print(image.RasterCount)
    [cols, rows] = band.shape

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(
        output_image,
        rows,
        cols,
        1,
        type,
    )
    # print(outdata)
    outdata.GetRasterBand(1).WriteArray(band)

    outdata.SetProjection(projection)  ##sets same projection as input
    outdata.SetGeoTransform(new_transformation)  ##sets same geotransform as input

    outdata.FlushCache()  ##saves to disk!!
    outdata = None


def ShapeFile2Raster(
    InputVector, OutputImage, RefImage, attribut, type=gdal.GDT_Float64
):
    """
        Fonction de rasterisation
    :param InputVector: shapefile à rasteriser
    :param OutputImage: image de sortie
    :param RefImage: image de reference qui est geo-referencé
    :param attribut: champs de valeur à rasteriser dans le shapefile
    :param type: type de l'image en sortie
    :return:
    """

    gdalformat = "GTiff"

    ##########################################################
    # Get projection info from referenced image
    image = gdal.Open(RefImage, gdal.GA_ReadOnly)

    # Open Shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    Shapefile = driver.Open(InputVector, 0)
    Shapefile_layer = Shapefile.GetLayer(0)

    # Rasterise
    print("Rasterising shapefile...")
    driver2 = gdal.GetDriverByName("GTiff")

    Output = driver2.Create(OutputImage, image.RasterXSize, image.RasterYSize, 1, type)
    Output.SetProjection(image.GetProjectionRef())
    Output.SetGeoTransform(image.GetGeoTransform())

    Band = Output.GetRasterBand(1)
    Band.SetNoDataValue(0)

    err = gdal.RasterizeLayer(
        Output, [1], Shapefile_layer, options=["ATTRIBUTE=" + attribut]
    )  # options=["ALL_TOUCHED=TRUE","ATTRIBUTE=Landscape"] burn_values= [255, 0,0]
    if err != 0:
        raise Exception("error rasterizing layer: %s" % err)

    Output.FlushCache()

    Band = None
    Output = None
    image = None
    Shapefile = None

    # Build image overviews
    # subprocess.call("gdaladdo --config COMPRESS_OVERVIEW DEFLATE " + OutputImage + " 2 4 8 16 32 64", shell=True)

    print("Done.")
