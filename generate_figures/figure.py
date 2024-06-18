import matplotlib.pyplot as plt

from eo_master2 import tools


def crop(image, x, y, width, height):

    crop = image[x : x + width, y : y + height, :]

    return crop


def plot_profiles(sits, gt_image, coords, output_figure):

    profiles = sits[coords]
    classes = gt_image[coords]

    plt.figure()

    for prof, c in zip(profiles, classes):
        plt.plot(prof, c=c)

    plt.tight_layout()
    plt.savefig(output_figure)


ImportWarning

classes = np.unique(img.ravel())

coordinates = {}
for c in classes:
    coords = np.argwhere(classes == c)
    idx = np.random.randint(0, len(coords))
    coord = coords[idx]
    coordinates[c] = coord

with open(...) as f:
    pkl.dump(f, coordinates)

{
    "code_class": 0,
    "coordinates": [x, y]
    "time_series": [....] => 187
}