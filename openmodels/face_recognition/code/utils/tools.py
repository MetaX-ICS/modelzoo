

import cv2
import json


Jsonfile = "./%s/config.json"

def get_params(modelname):
    
    with open(Jsonfile%(modelname), 'r') as f:
        param_all = json.load(f)
        params = param_all["model_config"]
    return params#,commons_file

def preprocess_test(path):#,commons_file
    image_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            image_list.append(line.strip('\n').split())
    return image_list


def write_dataset(filename):
    coco_val = COCO(filename)
    imgIds = sorted(coco_val.getImgIds())
    img_list = coco_val.loadImgs(imgIds)
    with open('./data/dataset.txt', "w") as f:
        for img in img_list:
            f.write("val2017/{} {} {} {}\n".format(img['file_name'],img['id'],img['width'], img['height']))


if __name__ == '__main__':
    test(".")