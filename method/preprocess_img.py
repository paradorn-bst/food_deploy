from PIL import Image
import numpy as np
from numpy import asarray
import json
import warnings
from io import BytesIO
import base64
import time
import onnxruntime
import onnx
import tf2onnx

#from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')



def crop_image(img):

    # Get image width and height
    width, height = img.size

    # If width and height is same length of pixel
    if width == height:
        # return image
        return img

    # Change data type to numpy array
    img = np.array(img)

    # Fiind half of offset between height and width
    offset  = int(abs(height-width)/2)

    # If width > height crop the offset of width
    if width > height:
        img = img[:,offset:(width-offset),:]
    # Else crop the offset of height
    else:
        img = img[offset:(height-offset),:,:]

    # return image
    return Image.fromarray(img)


def preprocess_img(img_path):

    # Try to open image
    try:
        # Open image from base64 format and convert to RGB
        im = Image.open(BytesIO(base64.b64decode(img_path))).convert('RGB')
        # Crop image method
        im = crop_image(im)
        # Resize image to 224 * 224
        im = im.resize((224, 224))
        # Convert to numpy array
        im = np.array(asarray(im))
        # Reshape to 4 dimension numpyarray
        im = im.reshape(-1, im.shape[0], im.shape[1], im.shape[2])
        # return image
        return im.astype('float32')

    # If reeturn string error when file type not support
    except:
        return "Your image file type does not support. Please select another image"