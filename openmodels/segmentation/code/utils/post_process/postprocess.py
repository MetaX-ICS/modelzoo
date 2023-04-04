import time
import os
import sys

import numpy as np

from tensorflow.keras.utils import to_categorical

from .yolact_mask import get_yolact_mask
from .deeplabv3_mask import get_deeplabv3_mask
from .solo_mask import get_solo_mask

def get_masks(type, outputs, batch_size, params, content):
    content_list = ["get_deeplabv3_mask", "get_solo_mask"]
    if params["post_process"] not in content_list:
        return eval(type)(outputs, batch_size, params)
    elif params["post_process"] == "get_deeplabv3_mask":
        return eval(type)(outputs, batch_size, params, content)
    elif params["post_process"] == "get_solo_mask":
        return eval(type)(outputs, batch_size, params, content)
    else:
        pass


def get_unet_mask(outputs, batch_size, params):
    channel, height, width = params["input_size"].split(",")

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]

    npreds = []
    for idx in range(batch_size):
        output = outputs[0][idx]
        ## one_hot
        masks = np.expand_dims(np.argmax(output, axis=0), axis=-1)   
        masks = np.transpose(to_categorical(masks,2), (2, 0, 1))

        npreds.append(masks)

    return npreds

def get_voc_mask(outputs, batch_size, params):
    channel, height, width = params["input_size"].split(",")

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]

    npreds = []
    for idx in range(batch_size):
        output = outputs[0][idx]
        ## one_hot
        # masks = np.expand_dims(np.argmax(output, axis=0), axis=-1)   
        masks = output.squeeze().astype(np.int64)

        npreds.append(masks)

    return npreds

def get_enet_mask(outputs, batch_size, params):
    channel, height, width = params["input_size"].split(",")

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]

    npreds = []
    for idx in range(batch_size):
        output = outputs[0][idx]
        ## one_hot
        # masks = np.expand_dims(np.argmax(output, axis=0), axis=-1)   
        masks = output.astype(np.float32)

        npreds.append(masks)

    return npreds

def get_mmdep_mask(outputs, batch_size, params):
    channel, height, width = params["input_size"].split(",")

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]

    npreds = []
    for idx in range(batch_size):
        output = outputs[0][idx]
        ## one_hot
        # masks = np.expand_dims(np.argmax(output, axis=0), axis=-1)   
        masks = output[0]

        npreds.append(masks)

    return npreds

def get_erfnet_mask(outputs, batch_size, params):
    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].copy().reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]

    npreds = []
    for idx in range(batch_size):
        output = outputs[0][idx].copy()
        masks = np.argmax(output, axis=0)
        
        npreds.append(masks)
    return npreds
