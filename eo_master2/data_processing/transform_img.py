from eo_master2 import tools

# Coordonnées et dimensions de la région à recadrer
image_filepath="C:/Users/Anis Speiler/Desktop/Master2 MEMOIRE/T31TCJ_Toulouse_Nord/20230101.tif"
x1=100
y1=200
x2=300
y2=400
output_folder="C:/Users/Anis Speiler/Desktop/Master2 MEMOIRE/"

# Recadrer l'image
tools.crop_geoimage(image_filepath,x1,y1,x2,y2,output_folder)

img,_,_ = tools.readImage('C:/Users/Anis Speiler/Desktop/Master2 MEMOIRE/20230101.tif')


newImg = tools.toNDVI(img)
newImg = ((newImg - newImg.min()) /( newImg.max() - newImg.min())) * 255
tools.saveImage(newImg, 'C:/Users/Anis Speiler/Desktop/Master2 MEMOIRE/ndvi.JPG')


newImg = tools.toNDWI(img)
newImg = ((newImg - newImg.min()) /( newImg.max() - newImg.min())) * 255
tools.saveImage(newImg, 'C:/Users/Anis Speiler/Desktop/Master2 MEMOIRE/ndwi.JPG')

newImg = tools.toIB(img)
newImg = ((newImg - newImg.min()) /( newImg.max() - newImg.min())) * 255
tools.saveImage(newImg, 'C:/Users/Anis Speiler/Desktop/Master2 MEMOIRE/ib.JPG')


nirRG = img[: , : , 0 : 3]
nirRG = tools.normalisation(nirRG) * 255
tools.saveImage(nirRG , 'C:/Users/Anis Speiler/Desktop/Master2 MEMOIRE/nirRg.JPG')

rgb = img[: , : , 1 : 4]
rgb = tools.normalisation(rgb) * 255 
tools.saveImage(rgb , 'C:/Users/Anis Speiler/Desktop/Master2 MEMOIRE/rgb.JPG')