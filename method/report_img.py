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


def result_to_json(first, another):

    # create dict
    result_dict = {}

    # if lenghth of forst and another not 0
    if len(first) and len(another) != 0:
        # result status is success
        result_dict['status'] = 'succcess'
    else:
        # result status fail
        result_dict['status'] = 'fail'

    # add first dict to result
    result_dict['result'] = {'food_type' : list(first.keys())[0], 'confidence' : list(first.values())[0]}

    # create dict in another result
    result_dict['another_result'] = dict()

    # loop to another food class
    for i in range(len(another)):

        # set new key
        key = f'food_another_type_{i}'

        # set new value
        val = f'food_another_confidence_{i}'

        # add dict to another result
        result_dict['another_result'].update(dict({key: list(another.keys())[i], val : list(another.values())[i]}))

    # dump result dict to json
    result_json = json.dumps(str(result_dict))

    # return json format result
    return result_json