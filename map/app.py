from flask import Flask, render_template
from flask import jsonify, send_from_directory, send_file, url_for

import io
import base64
from PIL import Image

from map.utils import tools

app = Flask(__name__)

# IMAGE_DIRECTORY = "/media/mohamed/Data/TeleDetection/DATA"
IMAGE_DIRECTORY = "static/map"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/load_image", methods=["GET"])
def load_image():
    tile = "T31TCJ_Toulouse_Nord"
    classes_file = "%s/%s_Classe.tif" % (IMAGE_DIRECTORY, tile)

    img, trans, proj = tools.readImage(classes_file)
    print(img.shape)
    height, width, dim = img.shape
    minx = trans[0]
    miny = trans[3] + width * trans[4] + height * trans[5]
    maxx = trans[0] + width * trans[1] + height * trans[2]
    maxy = trans[3]

    buffered = io.BytesIO()
    image = Image.fromarray(img[:, :, 0])
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify(array=encoded_image, minx=minx, miny=miny, maxx=maxx, maxy=maxy)


@app.route("/zoom/<string:city>")
def zoom(city):
    # latlong = tools.getLatitudeLongitude(f"{IMAGE_DIRECTORY}/{filename}")
    geodata = {
        "Toulouse": [43.598295, 1.41449],
        "Bejaya": [36.747413, 5.056458],
        "BordjBouArreridj": [36.069776, 4.759827],
    }
    print(geodata[city])
    if city in geodata.keys():
        # Returning the geographic information as JSON
        return jsonify(
            {
                "lattitude": geodata[city][0],
                "longitude": geodata[city][1],
            }
        )
    else:
        return jsonify({"error": "The selected city is not available."})


@app.route("/overlay_landcover/<string:city>")
def overlay_landcover(city):
    landcover = {
        "Toulouse": "Toulouse_landcover.png",
        "Bejaya": "Bejaya_landcover.png",
        "Bordj Bou Arreridj": "BBA_landcover.png",
    }

    print(landcover[city])

    if not landcover[city] is None:
        print("=============")
        tl_latlong, tr_latlong, bl_latlong, br_latlong = tools.getLatitudeLongitude(
            f"map/{IMAGE_DIRECTORY}/{landcover[city]}"
        )
        print(tl_latlong, tr_latlong, bl_latlong, br_latlong)
        # print("===> ", send_from_directory(IMAGE_DIRECTORY, landcover[city]))
        # Returning the geographic information as JSON
        return jsonify(
            {
                "tl_latlong": tl_latlong,
                "tr_latlong": tr_latlong,
                "bl_latlong": bl_latlong,
                "br_latlong": br_latlong,
            }
        )
    else:
        return jsonify(
            {"error": "Failed to read geographic information from the image"}
        )

    # return send_from_directory(IMAGE_DIRECTORY, filename)


@app.route("/send_image/<string:city>")
def send_image(city):
    landcover = {
        "Toulouse": "Toulouse_landcover.png",
        "Bejaya": "Bejaya_landcover.png",
        "Bordj Bou Arreridj": "BBA_landcover.png",
    }
    return send_from_directory(
        IMAGE_DIRECTORY, landcover[city]
    )  # f"{IMAGE_DIRECTORY}/{landcover[city]}")
    # return url_for(f"{IMAGE_DIRECTORY}/{landcover[city]}")


if __name__ == "__main__":
    app.run(debug=True)
