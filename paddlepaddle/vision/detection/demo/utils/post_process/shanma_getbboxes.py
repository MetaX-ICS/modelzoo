from math import sqrt, ceil
from itertools import product
import numpy as np

from .nms import nms_bboxes, filter_bbox, sigmoid, softmax

def _make_grid(featmap_size, stride):
    feat_h, feat_w = featmap_size
    shift_x = np.arange(0, feat_w) * stride
    shift_y = np.arange(0, feat_h) * stride
    xv, yv = np.meshgrid(shift_x, shift_y)
    xv = xv.flatten()
    yv = yv.flatten()
    cx = xv + stride//2
    cy = yv + stride//2
    return np.stack((cx, cy), axis=-1)

def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def get_shanma_bboxes(outputs, batch_size, params):
    channel, width, height = params["input_size"].split(",")
    
    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]

    strides = (4, 16, 32, 64)
    mlvl_anchors = []
    input_shape = (int(width), int(height))
    for i in range(len(strides)):
        anchors = _make_grid((ceil(input_shape[0]/strides[i]), ceil(input_shape[1]/strides[i])), strides[i])
        mlvl_anchors.append(anchors)

    npreds = []
    for idx in range(batch_size):
        bboxes, classes, scores = [], [] ,[]
        for i in range(len(strides)):
            stride = strides[i]
            anchors = mlvl_anchors[i]
            bbox_cls = outputs[i][idx][0]
            bbox_pred = outputs[4+i][idx][0]
            bbox_confs = outputs[8+i][idx][0]

            bbox = distance2bbox(anchors, bbox_pred*stride, input_shape)
            bbox_xy = bbox[...,:2] / (int(width), int(height))
            bbox_wh = (bbox[...,2:4] - bbox[...,:2]) / (int(width), int(height))
            bbox = np.concatenate((bbox_xy, bbox_wh), axis=-1)

            bboxs, classs, score = filter_bbox(bbox, bbox_confs, bbox_cls, 0.01)

            bboxes.append(bboxs)
            classes.append(classs)
            scores.append(score)

        bboxes = np.concatenate(bboxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        nbboxes, nclasses, nscores = [], [], []
        for cl in set(classes):
            inds = np.where(classes == cl)
            b = bboxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = nms_bboxes(b, s, 0.5)

            nbboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if len(nclasses) != 0 and len(nscores) != 0:
            nbboxes = np.concatenate(nbboxes)
            nclasses = np.concatenate(nclasses)
            nscores = np.concatenate(nscores)
            preds = np.concatenate((nbboxes, np.expand_dims(nclasses, axis=-1), np.expand_dims(nscores, axis=-1)), axis=-1)
            npreds.append(preds)
        else:
            npreds.append(np.array([]))

    return npreds