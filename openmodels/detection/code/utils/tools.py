
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import sys
import cv2
import json

coco80_to_coco91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 
56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 
67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'] 

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



def test_mAP(modelname,batchsize,precision):
    cocoGt=COCO("./data/instances_val2017.json")

    pred_jsonpath = modelname + "/val2017_predictions.json"  # predictions json
    cocoDt=cocoGt.loadRes(pred_jsonpath)

    imgIds=sorted(cocoDt.getImgIds())
    cocoEval = COCOeval(cocoGt,cocoDt,"bbox")
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

       
def test(modelname):
    
    with open("./predictions_52d4a4aa.json", "r") as f:
        jdict = json.load(f)
    cocoGt=COCO("./data/instances_val2017.json")
    imgIds=sorted(cocoGt.getImgIds())[:100]

    for img_info in cocoGt.loadImgs(imgIds):
        # img_info = img_infos[0]
        bboxes = [x for x in jdict if x["image_id"]==img_info["id"]]
        img = cv2.imread("./data/val2017/"+img_info['file_name'])
        for bbox in bboxes:
            if bbox['score'] > 0.3:
                cv2.rectangle(img, (int(bbox["bbox"][0]), int(bbox["bbox"][1])), (int(bbox["bbox"][0]+bbox["bbox"][2]), int(bbox["bbox"][1]+bbox["bbox"][3])), (255, 0, 0), 2)
                cv2.putText(img, '{0} {1:.2f}'.format(names[coco80_to_coco91.index(bbox["category_id"])], bbox['score']),
                            (int(bbox["bbox"][0]), int(bbox["bbox"][1]+20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,0,255))

        cv2.imwrite(modelname + "/reslut.jpg", img) 

def test_voc(modelname,batchsize,precision):
    
    with open(modelname + "/voc_results/det_test_bicycle.txt", "r") as f:
        for line in f.readlines():
            # img_info = img_infos[0]
            img_info = line.strip().split(" ")
            score = img_info[1]
            bbox = img_info[2:]
            img = cv2.imread("./data/JPEGImages/"+img_info[0]+".jpg")
            if float(score) > 0.3:
                cv2.rectangle(img, (int(float(bbox[0])), int(float(bbox[1]))), (int(float(bbox[2])), int(float(bbox[3]))), (255, 0, 0), 2)
                cv2.putText(img, 'bicycle {}'.format(score),
                            (int(float(bbox[0])), int(float(bbox[1])+20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,0,255))

            cv2.imwrite(modelname + "/reslut.jpg", img) 

def rewrite_voctxt(filename="test_voc.txt"):
    file_list = []
    with open("./data/" + filename, "r") as f:
        for line in f.readlines():
            # filename = "./JPEGImages/" + line.strip() + ".jpg"
            # file_list.append(filename)
            file_list.append(line.strip())

    with open('./data/voc.txt', "w") as f:
        for filename in file_list:
            h, w, _ = cv2.imread("./data/JPEGImages/"+filename+".jpg").shape
            f.write("{} {} {} {}\n".format( "JPEGImages/"+filename+".jpg", filename, w, h))

def write_dataset(filename):
    coco_val = COCO(filename)
    imgIds = sorted(coco_val.getImgIds())
    img_list = coco_val.loadImgs(imgIds)
    with open('./data/dataset.txt', "w") as f:
        for img in img_list:
            f.write("val2017/{} {} {} {}\n".format(img['file_name'],img['id'],img['width'], img['height']))


if __name__ == '__main__':
    modelname = sys.argv[1]
    test(modelname)