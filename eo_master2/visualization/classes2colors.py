import numpy as np
from PIL import Image
import json

# Load TIF image
image = Image.open("T31TCJ_Toulouse_Nord_Classe.tif")
img_np = np.array(image)

# Load nomenclature data
with open('../constants/classes_labels.json', 'r') as json_file:
    data = json.load(json_file)
    categories = data['Level2']

# Create a dictionary lookup for pixel codes
code_to_rgb = {cat['code']: tuple(cat['RGB color']) for cat in categories.values()}

# Create an RGB image
rgb_image = Image.new("RGB", image.size)

# Iterate through each pixel
for x in range(image.width):
    for y in range(image.height):
        pixel_code = img_np[y][x]
        if pixel_code in code_to_rgb:
            rgb_image.putpixel((x, y), code_to_rgb[pixel_code])

# Save the modified image
output_path = "new.tif"
rgb_image.save(output_path)