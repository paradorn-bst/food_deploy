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


def predict_img(image, classifier, recommend = 5):

    # get class of dataset
    food_classes_dict = {0: 'FriedKale',
                         1: 'FriedMusselPancakes',
                         2: 'KaiJeowMooSaap',
                         3: 'KaoManGai',
                         4: 'KaoMooDang',
                         5: 'KaoMunGai-Tord',
                         6: 'KhanomJeenNamYaKati',
                         7: 'KhaoMokGai',
                         8: 'KhaoMooTodGratiem',
                         9: 'Khaopad',
                         10: 'Khaosoi',
                         11: 'Khua-Kai',
                         12: 'KkaoKlukKaphi',
                         13: 'KuayJab',
                         14: 'KuayTeowReua',
                         15: 'Padseiew',
                         16: 'PadThai',
                         17: 'PhatKaphrao',
                         18: 'PorkStickyNoodles',
                         19: 'StewedPorkLeg',
                         20: 'Suki',
                         21: 'Yentafo'}

    # get input and output name
    input_name = classifier.get_inputs()[0].name
    output_name = classifier.get_outputs()[0].name

    # predict image
    class_prob = classifier.run([output_name], {input_name: image})
    # change to numpy array
    class_prob = np.array(class_prob)

    # create dict to collect prob of class
    class_prob_dict = dict(enumerate(class_prob.flatten(), 0))

    # change number of key to foodclass
    class_prob_dict = {food_classes_dict[key] : val for key, val in class_prob_dict.items()}

    # extract most probability of food class
    class_prob_dict_first = sorted(class_prob_dict.items(), key=lambda x:x[1], reverse = True)[0:1]
    # save to dict
    class_prob_dict_first = dict(class_prob_dict_first)

    # extract another class prob
    class_prob_dict_another = sorted(class_prob_dict.items(), key=lambda x:x[1], reverse = True)[1: recommend]
    # save to dict
    class_prob_dict_another = dict(class_prob_dict_another)

    # return most prob dict and another prob dict
    return class_prob_dict_first, class_prob_dict_another