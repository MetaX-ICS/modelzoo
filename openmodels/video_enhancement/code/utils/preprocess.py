import numpy as np
import cv2
import numbers
import os

from torchvision import transforms
from PIL import Image


def base_preprocess(image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=(224,224), reverse=False):
    image = cv2.imread(image_path)
    if reverse:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # imgcrop = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    imgcrop = image.astype(np.float32)
    imgcrop -= mean
    imgcrop /= std

    imgcrop = imgcrop.transpose([2, 0, 1])
    return imgcrop

def vdsr_preprocess(image_file, mean, std, size, reverse=False):
    image = np.load(image_file)

    image = image.astype(np.float32)
    imgcrop = np.expand_dims(image, axis=-1)
    imgcrop -= mean[0]
    imgcrop /= std[0]
    imgcrop = imgcrop.transpose([2, 0, 1])
    return imgcrop

def get_images(image_list, params):
    # The mean, std value is in the prototxt, if not, default is 0.0/1.0
    if "scale" in params.keys():
        std = tuple([ 1/float(scale) for scale in params['scale'].split(",") ])
    elif "std" in params.keys():
        std = tuple([float(scale) for scale in params['std'].split(",") ])
    else:
        std = 1.0
    
    if "mean" in params.keys():
        mean = tuple([float(mean) for mean in params['mean'].split(",")]) 
    else:
        mean = 0.

    reverse = True if "reverse_channel" in params.keys() else False
    size= [int(s) for s in params["input_size"].split(",")]
    
    if "pre_process" in params.keys():
        preprocess = eval(params["pre_process"])
    else:
        preprocess = base_preprocess

    dir_pre = os.path.split(params["image_T4"])[0]+"/"
    inputs = [preprocess(dir_pre+im_f[0],  mean, std, (size[1],size[2]), reverse) for im_f in image_list]
    input_ = np.zeros((len(image_list), *size), dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        input_[ix] = in_
    return input_
