import numpy as np
import cv2
import numbers
import os

from torchvision import transforms
from PIL import Image


def yolox_preprocess(image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=(224,224), reverse=False, swap=(2, 0, 1)):
    img = cv2.imread(image_path)
    if len(img.shape) == 3:
        padded_img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(size, dtype=np.uint8) * 114

    r = min(size[0] / img.shape[0], size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img#, r

def yolov5_preprocess(image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=(224,224), reverse=False, swap=(2, 0, 1)):
    img = cv2.imread(image_path)
    if len(img.shape) == 3:
        padded_img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(size, dtype=np.uint8) * 114

    r = min(size[0] / img.shape[0], size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        # interpolation=cv2.INTER_LINEAR,
        interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)[::-1]
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)/255.
    return padded_img#, r

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

def rfb_preprocess(image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=224, reverse=False):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    img -= mean

    img = img.transpose((2,0,1))
    return img

def effc_preprocess(image_path,mean=(103.53,116.28,123.675), std=(57.375,57.12,58.395),  input_size=(1280,1280), reverse=False):
    img = cv2.imread(image_path).astype(np.float32)
    img -= mean   
    img /= std

    if len(img.shape) == 3:
        padded_img = np.zeros((input_size[0], input_size[1], 3), dtype=np.float32)
    else:
        padded_img = np.ones(input_size, dtype=np.float32)

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        # interpolation=cv2.INTER_LINEAR,
        interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    swap=(2, 0, 1)

    padded_img = padded_img.transpose(swap)[::-1]
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    # np.savetxt("0139.txt",padded_img.reshape(-1),fmt="%18.15f")
    return padded_img

def fcos_preprocess(image_file, mean=(102.9801, 115.9465, 122.7717), std=(1.,1.,1.),input_size=(1333,1333),  reverse=False):
    # img = cv2.imread("./data/" + image_file).astype(np.float32)
    img = Image.open(image_file).convert("RGB")
    
    padded_img = np.zeros((3, input_size[0], input_size[1]), dtype=np.float32)

    r = min(input_size[0] / img.size[0], input_size[1] / img.size[1])
    preprocess = transforms.Compose([
        transforms.Resize((int(img.size[1] * r), int(img.size[0] * r))),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),
    ])

    preprocess_normalize = transforms.Compose(
        [
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # resized_img = preprocess(img).numpy() * 255.
    resized_img = preprocess(img)[[2, 1, 0]] * 255
    resized_img = preprocess_normalize(resized_img).numpy()
    # resized_img = cv2.resize(
    #     img,
    #     (int(img.shape[1] * r), int(img.shape[0] * r)),
    #     # interpolation=cv2.INTER_LINEAR,
    #     interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR,
    # )
    padded_img[:,: int(img.size[1] * r), : int(img.size[0] * r)] = resized_img

    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    # np.savetxt("0139.txt",padded_img.reshape(-1),fmt="%18.15f")
    return padded_img#, r

def mmdep_preprocess(image_file, mean=(102.9801, 115.9465, 122.7717), std=(1.,1.,1.),input_size=(608,608),  reverse=False):
    # img = cv2.imread("./data/" + image_file).astype(np.float32)
    img = Image.open(image_file).convert("RGB")
    
    padded_img = np.zeros((3, input_size[0], input_size[1]), dtype=np.float32)

    r = min(input_size[0] / img.size[0], input_size[1] / img.size[1])
    preprocess = transforms.Compose([
        transforms.Resize((int(img.size[1] * r+0.5), int(img.size[0] * r+0.5))),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    resized_img = preprocess(img).numpy()

    padded_img[:,: int(img.size[1] * r+0.5), : int(img.size[0] * r+0.5)] = resized_img#[[2, 1, 0]]

    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img

def shanma_preprocess(image_file, mean=(102.9801, 115.9465, 122.7717), std=(1.,1.,1.),input_size=(608,608),  reverse=False):
    img = cv2.imread(image_file)#.astype(np.float32)
    
    padded_img = np.zeros((input_size[0], input_size[1], 3), dtype=np.float32)

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    new_size = (int(img.shape[1] * r), int(img.shape[0] * r))
    resized_img = cv2.resize(img, new_size)

    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img /= 255

    padded_img = (padded_img-mean) / std
    padded_img = padded_img.transpose((2,0,1))
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img

def retina_preprocess(image_file, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), input_size=(480,640), reverse=False):
    img = Image.open(image_file).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    resized_img = preprocess(img).numpy()

    return resized_img

def ppyolo_preprocess(image_file, mean=(102.9801, 115.9465, 122.7717), std=(1.,1.,1.),input_size=(608,608),  reverse=False):
    img = cv2.imread(image_file)[:,:,[2,1,0]]
    
    padded_img = cv2.resize(img, input_size, interpolation=2).astype(np.float32)

    padded_img /= 255.

    padded_img = (padded_img-mean) / std
    padded_img = padded_img.transpose((2,0,1))
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img

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
