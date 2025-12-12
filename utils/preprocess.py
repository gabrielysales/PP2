import numpy as np
from PIL import Image

def preprocess_image(image, target_size=(150, 150)):
    image = image.resize(target_size)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # canal grayscale
    img_array = np.expand_dims(img_array, axis=0)   # batch
    return img_array
