import numpy as np
import time
import os
import sys
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import xml.etree.ElementTree as ET

from .yolo_getbboxes import get_yolov4_bboxes_v1, get_yolov3_bboxes_v1, get_yolov2_bboxes, get_yolov3_tiny_bboxes_v1, \
    get_yolov4_tiny_bboxes, get_yolox_bboxes, get_yolov5_bboxes, get_yolov5_bboxes_v1, get_yolov3_mmdeploy_postprocess, \
    get_yolov7_bboxes_v1, get_yolov3_pd_bboxes, get_ppyoloe_pd_bboxes
from .ssd_getbboxes import get_mobilenetv1_ssd_bboxes_v1, get_vgg16_ssd_bboxes_v1, get_resnet34_ssd_bboxes
from .rfb_getbboxes import get_rfb_bboxes
from .effc_getbboxes import get_effc_bboxes
from .fcos_getbboxs import get_fcos_bboxes
from .retina_getbboxes import get_retina_bboxes, get_retina_bboxes_v1, get_retina_pad_bboxes
from .shanma_getbboxes import get_shanma_bboxes


coco80_to_coco91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 
56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 
67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def get_bboxes(type, outputs, batch_size, params, start_idx):
    if "vgg16_ssd" in type: 
        return eval(type)(outputs, batch_size, params, start_idx)
    else:
        return eval(type)(outputs, batch_size, params)

def comute_mAP_coco(preds, params, save_path, iters):
    label_file = params['label_list']
    is91 = True if params['is91'] == "True" else False
    cocoGt=COCO(label_file)

    inputs = preprocess_test(params['image_T4'])
    jdict = []
    for pred, input in zip(preds, inputs):
        if len(pred) != 0:
            save_one_json(pred, jdict, input[1], input[2:4], is91, params["post_process"])

    pred_jsonpath = save_path + "/val2017_predictions.json"  # predictions json
    with open(pred_jsonpath, 'w') as f:
        json.dump(jdict, f)

    cocoDt=cocoGt.loadRes(pred_jsonpath)
    imgIds=sorted(cocoDt.getImgIds())
    cocoEval = COCOeval(cocoGt,cocoDt,"bbox")
    cocoEval.params.imgIds  = imgIds[:iters]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    mAP, mAP50 = cocoEval.stats[:2]
    
    return mAP, mAP50

def comute_mAP_voc(preds, params, save_path, iters):
    label_file = params['label_list']
    class_names = [name.strip() for name in open(label_file).readlines()]
    class_dict = {class_name: i for i, class_name in enumerate(class_names)}

    inputs = preprocess_test(params['image_list'])[:iters]
    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(inputs, class_dict)

    results = np.concatenate(preds, axis=0)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background

        if not os.path.exists(save_path+"/voc_results"):
            print("create a result path")
            os.mkdir(save_path+"/voc_results")

        prediction_path = save_path+"/voc_results/det_test_{}.txt".format(class_name)
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.shape[0]):
                prob_box = sub[i, 2:]
                # image_id = dataset.ids[int(sub[i, 0])]
                _, image_id, w, h = inputs[int(sub[i, 0])]
                size = np.array([float(w), float(h), float(w), float(h)])
                prob_box[1:] = prob_box[1:]*size + 1
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )

    aps = []
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0 or class_index not in true_case_stat.keys():
            continue
        prediction_path = save_path+"/voc_results/det_test_{}.txt".format(class_name)
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path
        )
        aps.append(ap)
        print(f"{class_name}: {ap}")

    mAP50 = sum(aps)/len(aps)
    print(f"\nAverage Precision Across All Classes:{mAP50}")
    return mAP50

### coco test
###
def _xywh2xywh(bbox, size):    
    ### clip  需要做截断
    y = bbox *size
    y[:, 0] = y[:,0].clip(0., size[0])
    y[:, 1] = y[:,1].clip(0., size[1])
    y[:, 2] = (y[:,2]+y[:,0]).clip(0., size[0]) - y[:,0] + 1
    y[:, 3] = (y[:,3]+y[:,1]).clip(0., size[1]) - y[:,1] + 1
    return y

def xywh2xywh(bbox, size):    
    ### clip  需要做截断
    y = np.copy(bbox)
    y[:, 0] = bbox[:,0].clip(0., 1.)
    y[:, 1] = bbox[:,1].clip(0., 1.)
    y[:, 2] = (bbox[:,2]+bbox[:,0]).clip(0., 1.) - y[:,0]
    y[:, 3] = (bbox[:,3]+bbox[:,1]).clip(0., 1.) - y[:,1] 

    return y*size

def xyxy2xywh(bbox, size):    
    ### clip  需要做截断
    y = np.copy(bbox)
    y[:, 0] = bbox[:,0].clip(0., 1.)
    y[:, 1] = bbox[:,1].clip(0., 1.)
    y[:, 2] = bbox[:,2].clip(0., 1.) - y[:,0]
    y[:, 3] = bbox[:,3].clip(0., 1.) - y[:,1] 

    return y*size

def save_one_json(predn, jdict, image_id, image_size, is91, post_name=None):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    pad_post_list = ["get_yolov7_bboxes_v1","get_yolov5_bboxes_v1","get_yolov5_bboxes","get_yolox_bboxes", \
        "get_effc_bboxes", "get_fcos_bboxes", "get_shanma_bboxes", "get_retina_bboxes_v1", \
            "get_yolov3_mmdeploy_postprocess", "get_retina_pad_bboxes"]#
    w, h = image_size
    if post_name in pad_post_list:
        size = [max(float(w), float(h))]*4
        box = xywh2xywh(predn[:, :4], size)
    elif "_pd_" in post_name:
        size = np.array([float(w), float(h), float(w), float(h)])
        box = xyxy2xywh(predn[:, 2:], size)
        predn[:,4] = predn[:,0]
        predn[:,5] = predn[:,1]
    else:
        size = np.array([float(w), float(h), float(w), float(h)])
        box = xywh2xywh(predn[:, :4], size)  # xywh 
    # box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': int(image_id),
                      'category_id': int(p[4]) if is91 else coco80_to_coco91[int(p[4])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[5], 5)})



### VOC test
###
def get_annotation(image_id, class_dict):
    annotation_file = "./data/Annotations/{}.xml".format(image_id)
    objects = ET.parse(annotation_file).findall("object")
    boxes = []
    labels = []
    is_difficult = []
    for object in objects:
        class_name = object.find('name').text.lower().strip()
        # we're only concerned with clases in our list
        if class_name in class_dict:
            bbox = object.find('bndbox')

            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])

            labels.append(class_dict[class_name])
            is_difficult_str = object.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

    return (np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            np.array(is_difficult, dtype=np.uint8))

def group_annotation_by_class(inputs, class_dict):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(inputs)):
        _, image_id, _, _ = inputs[i]
        annotation = get_annotation(image_id, class_dict)
        gt_boxes, classes, is_difficult = annotation
        # gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = np.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = np.array(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases

def compute_average_precision(precision, recall):
    """
    It computes average precision based on the definition of Pascal Competition. It computes the under curve area
    of precision and recall. Recall follows the normal definition. Precision is a variant.
    pascal_precision[i] = typical_precision[i:].max()
    """
    # identical but faster version of new_precision[i] = old_precision[i:].max()
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # find the index where the value changes
    recall = np.concatenate([[0.0], recall, [1.0]])
    changing_points = np.where(recall[1:] != recall[:-1])[0]

    # compute under curve area
    areas = (recall[changing_points + 1] - recall[changing_points]) * precision[changing_points + 1]
    return areas.sum()

def compute_voc2007_average_precision(precision, recall):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.
    return ap

def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, 1e5)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):

    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold = 0.5, use_2007_metric = True):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = np.expand_dims(np.array([float(v) for v in t[2:]]),axis=0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = iou_of(box, gt_box)
            max_iou = np.max(ious).item()
            max_arg = np.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return compute_voc2007_average_precision(precision, recall)
    else:
        return compute_average_precision(precision, recall)

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





