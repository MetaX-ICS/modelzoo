import cv2
import numpy as np 
import math
import json

import torch
import torch.nn.functional as F


nms_conf_thresh = 0.05
nms_thresh = 0.5
nms_top_k = 200
max_num_detections = 100

def decode(loc, priors):
    """
    Decode predicted bbox coordinates using the same scheme
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)

    """
    variances = [0.1, 0.2]
    
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    
    return boxes

def intersect(box_a, box_b):
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    return torch.clamp(max_xy - min_xy, min=0).prod(3)  # inter

def jaccard(box_a, box_b, iscrowd:bool=False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)

def fast_nms( boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]

    num_classes, num_dets = idx.size()

    boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = jaccard(boxes, boxes)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= iou_threshold)

    if second_threshold:
        keep *= (scores > nms_conf_thresh)

    # Assign each kept detection to its corresponding class
    classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    masks = masks[keep]
    scores = scores[keep]
    
    # Only keep the top cfg.max_num_detections highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    idx = idx[:max_num_detections]
    scores = scores[:max_num_detections]

    classes = classes[idx]
    boxes = boxes[idx]
    masks = masks[idx]

    return boxes, masks, classes, scores

def detect(conf_preds, decoded_boxes, mask_data):
    """ Perform nms for only the max scoring class that isn't background (class 0) """
    cur_scores = conf_preds[1:, :]
    conf_scores, _ = torch.max(cur_scores, dim=0)

    keep = (conf_scores > nms_conf_thresh)
    scores = cur_scores[:, keep]
    boxes = decoded_boxes[keep, :]
    masks = mask_data[keep, :]

    if scores.size(1) == 0:
        return None
    
    boxes, masks, classes, scores = fast_nms(boxes, masks, scores, nms_thresh, nms_top_k)

    return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

def get_yolact_mask(outputs, batch_size, params):
    channel, height, width = params["input_size"].split(",")

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]

    npreds = []
    for idx in range(batch_size):
        loc_data   = torch.from_numpy(outputs[0][idx].squeeze())
        conf_data  = torch.from_numpy(outputs[1][idx].squeeze())
        mask_data  = torch.from_numpy(outputs[2][idx].squeeze())
        prior_data = torch.from_numpy(outputs[3][idx].squeeze())
        proto_data = torch.from_numpy(outputs[4][idx].squeeze())


        num_priors = prior_data.size(0)
        conf_preds = conf_data.transpose(1, 0).contiguous()

        decoded_boxes = decode(loc_data, prior_data)
        result = detect(conf_preds, decoded_boxes, mask_data)
        if result is not None and proto_data is not None:
            result['proto'] = proto_data

        npreds.append(result)

    return npreds
