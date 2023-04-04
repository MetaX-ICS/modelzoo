import numpy as np
import collections
from itertools import product

from .nms import nms_bboxes, sigmoid, softmax

def filter_bbox(bboxes, bbox_cls, bbox_confs, thr = 0.01):

    per_candidate_inds = np.where(bbox_cls >= thr)
    # per_candidate_nonzeros = per_candidate_inds.nonzero()
    pos = per_candidate_inds[0]
    classes = per_candidate_inds[1]
    # # bbox_confs_score = bbox_confs * bbox_cls
    # classes = np.argmax(bbox_cls, axis=-1)
    # # bbox_class = np.expand_dims(bbox_class, axis=-1)
    # cls_scores = np.max(bbox_cls, axis=-1)

    # pos = np.where(cls_scores >= thr)
    bboxes = bboxes[pos]  
    # classes = classes[pos]
    scores = (bbox_cls*bbox_confs)[per_candidate_inds]

    return bboxes, classes, scores

def compute_locations_per_level(h, w, stride):
    shifts_x = np.arange(
        0, w * stride, step=stride,
        dtype=np.float32
    )
    shifts_y = np.arange(
        0, h * stride, step=stride,
        dtype=np.float32
    )
    shift_y, shift_x = np.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    # locations = np.stack((shift_x, shift_y), axis=1) + stride // 2
    locations = np.stack((shift_y, shift_x), axis=1) + stride // 2
    return locations

def compute_locations():
    features = [[167,167],[84,84],[42,42],[21,21],[11,11]]
    locations = []
    FPN_STRIDES = [8, 16, 32, 64, 128]
    for level, feature in enumerate(features):
        h, w = feature
        locations_per_level = compute_locations_per_level(
            h, w, FPN_STRIDES[level]
        )
        locations.append(locations_per_level)
    return locations

def get_fcos_bboxes(outputs, batch_size, params):

    channel, height, width = params["input_size"].split(",")

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs_size_list = np.array(outputs_size_list)[[0,10,5,1,11,6,2,12,7,3,13,8,4,14,9]]
    # outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]
    # anchors = Anchors((int(height), int(width)))
    anchors = compute_locations()

    npreds = []
    for idx in range(batch_size):
        bboxes, classes, scores = [], [] ,[]
        for i in range(len(outputs)//3):            
            bbox_cls = np.transpose(sigmoid(outputs[i][idx]),(1, 2, 0)).reshape(-1, 80)
            bbox_conf = np.transpose(sigmoid(outputs[i+10][idx]),(1, 2, 0)).reshape(-1, 1)
            bbox_reg = np.transpose(outputs[i+5][idx],(1, 2, 0)).reshape(-1, 4)
            anchor = anchors[i]

            # bbox = BBoxTransform(anchors, bbox).squeeze()
            bbox = np.stack([
                anchor[:, 0] - bbox_reg[:, 0],
                anchor[:, 1] - bbox_reg[:, 1],
                anchor[:, 0] + bbox_reg[:, 2],
                anchor[:, 1] + bbox_reg[:, 3],
            ], axis=1)
            bbox /= (int(width), int(height), int(width), int(height))
            
            bbox_xy = bbox[:,:2]
            bbox_wh = bbox[:,2:4] - bbox[:,:2]
            bbox = np.concatenate((bbox_xy, bbox_wh), axis=-1)

            bboxs, classs, score = filter_bbox(bbox, bbox_cls, bbox_conf, 0.05)

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

            keep = nms_bboxes(b, s)

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