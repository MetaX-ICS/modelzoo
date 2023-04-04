import numpy as np
import cv2
import numbers
import os

from PIL import Image
from torchvision import transforms


def base_preprocess(image_file, std, mean, size, reverse=False):
    image = cv2.imread(image_file)

    imgcrop = cv2.resize(image,size)
    imgcrop = imgcrop.astype(np.float32)

    imgcrop -= mean
    imgcrop /= std
    imgcrop = imgcrop.transpose([2, 0, 1])
    if reverse:
        imgcrop = imgcrop[[2,1,0]]
    return imgcrop

def yolact_preprocess(image_file, std, mean, size, reverse=False):
    image = cv2.imread(image_file)

    imgcrop = cv2.resize(image.astype(np.float32),size,)
    imgcrop = imgcrop

    imgcrop -= mean
    imgcrop /= std
    imgcrop = imgcrop.transpose([2, 0, 1])
    if reverse:
        imgcrop = imgcrop[[2,1,0]]

    return imgcrop

def unet_preprocess(image_file, std, mean, size, reverse=False):
    # image = cv2.imread("./data/" + image_file)
    image = Image.open(image_file)

    imgcrop = image.resize((size[1],size[0]))
    imgcrop = np.asarray(imgcrop).astype(np.float32)

    imgcrop -= mean
    imgcrop /= std
    imgcrop = imgcrop.transpose([2, 0, 1])
    return imgcrop

def enet_preprocess(image_file, std, mean, size, reverse=False):
    # image = cv2.imread("./data/" + image_file)
    img = Image.open(image_file).convert('RGB')
    short_size = size[0]
    w, h = img.size
    if w > h:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = short_size
        oh = int(1.0 * h * ow / w)

    img = img.resize((ow, oh), Image.BILINEAR)
    w, h = img.size
    x1 = int(round((w - short_size) / 2.))
    y1 = int(round((h - short_size) / 2.))
    img = img.crop((x1, y1, x1 + short_size, y1 + short_size))

    img = np.array(img).astype(np.float32)/255.

    img -= mean
    img /= std
    img = img.transpose((2,0,1))

    # img_bak = np.load("./data/Citysapes_data/Cityscapes_val/frankfurt_000001_068063_leftImg8bit.npy")
    return img

def voc_preprocess(image_file, std=[58.395, 57.12, 57.375], mean=[123.675, 116.28, 103.53], size=(500,500), reverse=False, swap=(2,0,1)):
    img = cv2.imread(image_file)
    if len(img.shape) == 3:
        padded_img = np.zeros((size[0], size[1], 3), dtype=np.float32)
    else:
        padded_img = np.zeros(size, dtype=np.float32) 

    r = min(size[0] / img.shape[0], size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        # interpolation=cv2.INTER_LINEAR,
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    if reverse:
        cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB,resized_img)
    resized_img -= mean
    resized_img /= std

    # pad_left = (size[1] - resized_img.shape[1]) // 2
    # pad_top = (size[0] - resized_img.shape[0]) // 2
    # padded_img[pad_top:resized_img.shape[0]+pad_top, pad_left:resized_img.shape[1]+pad_left] = resized_img
    padded_img[:resized_img.shape[0], :resized_img.shape[1]] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img#, r


def deeplabv3_preprocess(image_file,std=(0.229,0.224,0.225), mean=(0.485,0.456,0.406),  size=(640,640), reverse=False):
    img = Image.open(image_file).convert('RGB')

    ratio = max(img.size[1]/size[0], (img.size[0]/size[1]))
    resize_size = (int(img.size[1]/ratio), int(img.size[0]/ratio))
    padding = [0,0, size[1]-resize_size[1], size[0]-resize_size[0]]
    preprocess = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.Pad(padding, 0),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    resized_img = preprocess(img).numpy()
    return resized_img

def erfnet_preprocess(image_file,std=(0.229,0.224,0.225), mean=(0.485,0.456,0.406),  size=(640,640), reverse=False):
    img = Image.open(image_file).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    resized_img = preprocess(img).numpy()
    return resized_img


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
    inputs = [preprocess(dir_pre+im_f[0], std, mean, (size[1],size[2]), reverse) for im_f in image_list]
    input_ = np.zeros((len(image_list), *size), dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        input_[ix] = in_
    return input_
