import os
import numpy as np
import cv2
import numbers

from torchvision import transforms
from PIL import Image

def resize(image, size, interpolation=cv2.INTER_LINEAR):
    assert isinstance(image, np.ndarray)

    if isinstance(size, int):
        h, w = image.shape[:2]
        if (w <= h and w == size) or (h <= w and h == size):
            return image
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(image, (ow, oh), interpolation=interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(image, (ow, oh), interpolation=interpolation)
    else:
        return cv2.resize(image, tuple(size[::-1]), interpolation=interpolation)


# crop
def crop(image, y, x, h, w):
    assert isinstance(image, np.ndarray)

    return image[y:y + h, x:x + w]


# center crop
def center_crop(image, size):
    assert isinstance(image, np.ndarray)

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))

    h, w = image.shape[:2]
    oh, ow = size
    x = int(round((w - ow) / 2.))
    y = int(round((h - oh) / 2.))

    return crop(image, y, x, oh, ow)

def base_preprocess(image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=224, reverse=False):
    ori_size = 256 if size[0] <= 256 else size[0]  
    preprocess = transforms.Compose([
        transforms.Resize(ori_size, Image.BILINEAR if ori_size<300 else Image.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    swap = [0,1,2]if reverse else [2,1,0] 
    img = preprocess(Image.open(image_path).convert("RGB")).numpy()[swap]
    return img

def pd_preprocess(image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=224, reverse=False):
    with open(image_path, "rb") as f:
        img = f.read()
        data = np.frombuffer(img, dtype="uint8")
        image = cv2.imdecode(data, 1)[:,:,[2,1,0]]

    ori_size = 256 if size[0] <= 256 else size[0]
    image = resize(image, ori_size, interpolation=1)
    image = center_crop(image, size[0]).astype(np.float32)

    image /= 255.
    image -= mean
    image /= std

    return image.transpose([2,0,1])


def mbv1_preprocess(image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=224, reverse=False):
    image =cv2.imread(image_path)
    if reverse:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    ori_size = 256 if size[0] <= 256 else size[0]
    image = resize(image, ori_size)
    image = center_crop(image, size[0]).astype(np.float32)

    image -= mean
    image /= std

    return image.transpose([2,0,1])

def cifar_preprocess(image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=224, reverse=False):
    image =cv2.imread(image_path)

    if reverse:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = image.astype(np.float32)
    image -= mean
    image /= std

    return image.transpose([2,0,1])

def load_npy(image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=(224,224), reverse=False):
    
    swap = [0,1,2]if reverse else [2,1,0]
    image = np.load(image_path, allow_pickle=True)[swap]

    return image

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
    
    if "preprocess" in params.keys():
        preprocess = eval(params["preprocess"])
    else:
        preprocess = base_preprocess

    dir_pre = os.path.split(params["label_T4"])[0]+"/"
    images = [preprocess(dir_pre+im_f[0], mean, std, (size[1], size[2]), reverse) for im_f in image_list]

    input_ = np.zeros((len(image_list), *size), dtype=np.float32)
    for ix, in_ in enumerate(images):
        input_[ix] = in_

    label = [int(label_info[1]) for label_info in image_list]
    label = np.array(label)
    if 'cls_num' not in params.keys() or params['cls_num'].split(",")[0] == "1000":
        return input_, label-1
    elif "incev1" in params['model_file']:
        return input_, label-1
    else:
        return input_, label