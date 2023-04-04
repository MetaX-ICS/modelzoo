import numpy as np
import cv2
import numbers
import os

from PIL import Image
from torchvision import transforms

def pse_preprocess(image_file, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=(224,224), reverse=False):
    img = Image.open(image_file).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    resized_img = preprocess(img).numpy()
    return resized_img

def dbnet_preprocess(image_file, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=(224,224), reverse=False):
    with open(image_file, "rb") as f:
        img = f.read()
        img = np.frombuffer(img, dtype="uint8")
        img = cv2.imdecode(img, 1)

        src_h, src_w, _ = img.shape
        resize_h = size[0]
        resize_w = size[1]
        ori_h, ori_w = img.shape[:2]

        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))

        scale = np.float32(1./255.)
        shape = (1,1,3)
        mean = np.array(mean).reshape(shape).astype("float32")
        std = np.array(std).reshape(shape).astype("float32")

        img = (img.astype("float32") * scale -mean ) / std
        img = img.transpose([2,0,1])
        return img

def crnn_preprocess(image_file, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=(224,224), reverse=False):
    img = Image.open(image_file).convert("L")

    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    resized_img = preprocess(img).numpy()
    # comp = np.load("./data/IIIT5K_npys/train/1009_2.npy")
    return resized_img

def pdrec_preprocess(image_file, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=(224,224), reverse=False):
    image = cv2.imread(image_file)
    padding_img = np.zeros((3,size[0],size[1]),dtype=np.float32)
    if reverse:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h = image.shape[0]
    w = image.shape[1]

    ratio = w / float(h)
    if int(size[0] * ratio + 1) > size[1]:
        resize_w = size[1]
    else:
        resize_w = int(size[0] * ratio + 1)

    resized_img = cv2.resize(
        image,
        (resize_w, size[0]),
        interpolation=cv2.INTER_LINEAR,
        # interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR,
    )
    resized_img = resized_img.astype(np.float32)
    resized_img -= mean
    resized_img /= std
    resized_img = resized_img.transpose([2, 0, 1])

    padding_img[:,:, 0:resize_w] = resized_img
    padding_img = np.ascontiguousarray(padding_img, dtype=np.float32)
    return padding_img


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
