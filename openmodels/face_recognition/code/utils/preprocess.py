import os
import numpy as np
import cv2
import numbers

import imageio
from torchvision import transforms
from PIL import Image

def mob_preprocess(dir_pre, image_list, mean, std, size, reverse=False):
    folder_name = dir_pre + 'lfw-112X96'
    # nameLs = []
    # nameRs = []
    imgs = []
    for i, p in enumerate(image_list):
        # p = p.split('\t')
        if len(p) == 3:
            nameL = os.path.join(folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[2])))
        elif len(p) == 4:
            nameL = os.path.join(folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(folder_name, p[2], p[2] + '_' + '{:04}.jpg'.format(int(p[3])))
        # nameLs.append(nameL)
        # nameRs.append(nameR)
    # for 
        imgl = imageio.imread(nameL).astype(np.float32)
        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, 2)
        # imgr = scipy.misc.imread(self.imgr_list[index])
        imgr = imageio.imread(nameR).astype(np.float32)
        if len(imgr.shape) == 2:
            imgr = np.stack([imgr] * 3, 2)

        # imgl = imgl[:, :, ::-1]
        # imgr = imgr[:, :, ::-1]
        imglist = [imgl, imgl[:, ::-1, :], imgr, imgr[:, ::-1, :]]
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - mean) / std
            imglist[i] = imglist[i].transpose(2, 0, 1)      
        imgs.extend(imglist)
    return imgs

def face_preprocess(dir_pre, image_list, mean, std, size, reverse=False):
    folder_name = dir_pre + 'lfw_160x160'
    # nameLs = []
    # nameRs = []
    imgs = []
    for i, p in enumerate(image_list):
        # p = p.split('\t')
        if len(p) == 3:
            nameL = os.path.join(folder_name, p[0], p[0] + '_' + '{:04}.png'.format(int(p[1])))
            nameR = os.path.join(folder_name, p[0], p[0] + '_' + '{:04}.png'.format(int(p[2])))
        elif len(p) == 4:
            nameL = os.path.join(folder_name, p[0], p[0] + '_' + '{:04}.png'.format(int(p[1])))
            nameR = os.path.join(folder_name, p[2], p[2] + '_' + '{:04}.png'.format(int(p[3])))
        # nameLs.append(nameL)
        # nameRs.append(nameR)
    # for 
        imgl = imageio.imread(nameL).astype(np.float32)
        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, 2)
        # imgr = scipy.misc.imread(self.imgr_list[index])
        imgr = imageio.imread(nameR).astype(np.float32)
        if len(imgr.shape) == 2:
            imgr = np.stack([imgr] * 3, 2)

        # imgl = imgl[:, :, ::-1]
        # imgr = imgr[:, :, ::-1]
        imglist = [imgl, imgl[:, ::-1, :], imgr, imgr[:, ::-1, :]]
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - mean) / std
            imglist[i] = imglist[i].transpose(2, 0, 1)      
        imgs.extend(imglist)
    return imgs

def arc_preprocess(dir_pre, image_list, mean, std, size, reverse=False):
    imgs = []
    for image_file in image_list:
        image = cv2.imread(dir_pre + "lfw-128x128/" + image_file[0])
        if reverse: 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        image = np.dstack((image, np.fliplr(image))) 
        image = image.transpose((2, 0, 1))
        # image = image[np.newaxis, :, :, :]
        image = image.reshape((2, 3, size[0], size[1]))
        image = image.astype(np.float32, copy=False)
        image -= mean
        image /= std
        imgs.extend([image[0],image[1]])
    return imgs

def base_preprocess(image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=(224,224), reverse=False):
    image = cv2.imread(image_path)
    if reverse:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    imgcrop = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    imgcrop = imgcrop.astype(np.float32)
    imgcrop -= mean
    imgcrop /= std

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
    
    dir_pre = os.path.split(params["image_T4"])[0]+"/"
    if "pre_process" in params.keys():
        preprocess = eval(params["pre_process"])
        inputs = preprocess(dir_pre, image_list,  mean, std, (size[1],size[2]), reverse)
    else:
        preprocess = base_preprocess
        inputs = [preprocess(dir_pre+im_f[0],  mean, std, (size[1],size[2]), reverse) for im_f in image_list]
    
    input_ = np.zeros((len(inputs), *size), dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        input_[ix] = in_
    return input_
