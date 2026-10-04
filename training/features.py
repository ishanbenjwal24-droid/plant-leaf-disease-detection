"""Image features used by the classical ML baseline and Streamlit demo."""

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import hog


IMAGE_SIZE = (96, 96)


def extract_features(image):
    if isinstance(image, (str, bytes)):
        image = Image.open(image)
    image = ImageOps.exif_transpose(image).convert("RGB").resize(IMAGE_SIZE)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = rgb2gray(rgb)
    shape_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    hsv = rgb2hsv(rgb)
    color_features = np.concatenate(
        [hsv.mean(axis=(0, 1)), hsv.std(axis=(0, 1))]
    )
    return np.concatenate([shape_features, color_features]).astype(np.float32)
