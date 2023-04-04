import numpy as np
import collections
from itertools import product

from .nms import filter_bbox, nms_bboxes, sigmoid, softmax
import torch

def generate_anchors(strides, ratio_vals, scales_vals):
    'Generate anchors coordinates from scales/ratios'
    anchors = []
    for stride in strides:

        scales = np.array(scales_vals).repeat(len(ratio_vals), 0).reshape(9,1)
        # scales = np.transpose(scales,(0, 1)).view(-1, 1)
        ratios = np.array(ratio_vals * len(scales_vals))

        wh = np.array([stride]).repeat(len(ratios)*2).reshape(-1, 2)
        ws = np.sqrt(wh[:, 0] * wh[:, 1] / ratios)
        dwh = np.stack([ws, ws * ratios], axis=1)
        xy1 = 0.5 * (wh - dwh * scales)
        xy2 = 0.5 * (wh + dwh * scales)

        anchors.append(np.concatenate([xy1, xy2], axis=1))
        # anchors.append(np.concatenate([xy1[:,::-1], xy2[:,::-1]], axis=1))

    return anchors
    

def generate_anchors_v1(strides, ratio_vals, scales_vals):
    'Generate anchors coordinates from scales/ratios'
    anchors = []
    for stride in strides:
        scales = np.array(scales_vals)
        ratios = np.array(ratio_vals)

        h_ratios = np.sqrt(ratios)
        w_ratios = 1 / h_ratios
        ws = (stride * w_ratios[:,None]*scales[None, :]).reshape(-1)
        hs = (stride * h_ratios[:,None]*scales[None, :]).reshape(-1)

        x1 = -0.5 * ws[:,None]
        y1 = -0.5 * hs[:,None]
        x2 = 0.5 * ws[:,None]
        y2 = 0.5 * hs[:,None]

        

        anchors.append(np.concatenate([x1, y1, x2, y2], axis=1))
        # anchors.append(np.concatenate([xy1[:,::-1], xy2[:,::-1]], axis=1))

    return anchors

def generate_anchors_pd(strides, ratio_vals, scales_vals):
    'Generate anchors coordinates from scales/ratios'
    anchors = []
    for stride in strides:
        scales = np.array(scales_vals)
        ratios = np.array(ratio_vals)

        h_ratios = np.sqrt(ratios)
        w_ratios = 1 / h_ratios
        ws = (stride * scales[None, :] * w_ratios[:,None]).reshape(-1)
        hs = (stride * scales[None, :] * h_ratios[:,None]).reshape(-1)

        x1 = -0.5 * ws[:,None]
        y1 = -0.5 * hs[:,None]
        x2 = 0.5 * ws[:,None]
        y2 = 0.5 * hs[:,None]

        anchors.append(np.concatenate([x1, y1, x2, y2], axis=1))
        # anchors.append(np.concatenate([xy1[:,::-1], xy2[:,::-1]], axis=1))

    return anchors

def delta2box(deltas, anchors, size, stride):
    'Convert deltas from anchors to boxes'

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = np.exp(deltas[:, 2:]) * anchors_wh

    # return np.concatenate([pred_ctr - 0.5 * pred_wh,pred_ctr + 0.5 * pred_wh - 1], 1)
    return np.concatenate([pred_ctr - 0.5 * pred_wh,pred_wh], 1)

def decode(bbox_reg, bbox_cls, anchor, stride, output_size, top_k=1000, thr=0.05):
    num_anchors = anchor.shape[0] if anchor is not None else 1
    num_classes = output_size[0] // num_anchors
    height, width = output_size[-2:]

    # Keep scores over threshold
    keep = np.where(bbox_cls >= thr)
    if keep[0].size == 0:
        return np.array([]), np.array([]), np.array([])

    # Gather top elements
    scores = bbox_cls[keep]
    # scores, indices = np.topk(scores, min(top_k, keep.size()[0]), axis=0)
    indices = np.array(keep)[0]
    classes = (indices // width // height) % num_classes

    # Infer kept bboxes
    x = indices % width
    y = (indices // width) % height
    a = indices // num_classes // height // width
    box_head = bbox_reg.reshape(num_anchors, 4, height, width)
    
    boxes = box_head[a, :, y, x]

    grid = np.stack([x, y, x, y], 1) * stride + anchor[a, :]
    boxes = delta2box(boxes, grid, [width, height], stride)

    return boxes, classes, scores

def decode_v1(bbox_reg, bbox_cls, anchor, stride, output_size, top_k=1000, thr=0.05):
    max_score = bbox_cls.max(axis=-1)
    top_k = min(top_k, max_score.size)
    _, topk_inds = torch.from_numpy(max_score).topk(top_k)
    topk_inds = topk_inds.numpy()
    # _, indices = np.topk(max_cls, min(top_k, max_cls.size()[0]), axis=0)

    prior = anchor[topk_inds]
    bbox = bbox_reg[topk_inds]
    scores = bbox_cls[topk_inds]

    return bbox, scores, prior

def get_retina_bboxes(outputs, batch_size, params):

    channel, height, width = params["input_size"].split(",")

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs_size_list = np.array(outputs_size_list)[[5,6,7,8,9,0,1,2,3,4]]
    # outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]
    # anchors = Anchors((int(height), int(width)))
    anchors = generate_anchors([8,16,32,64,128], [1.0, 2.0, 0.5],[4 * 2 ** (i / 3) for i in range(3)])

    npreds = []
    for idx in range(batch_size):
        bboxes, classes, scores = [], [] ,[]
        for i in range(len(outputs)//2):            
            bbox_cls = outputs[i][idx].reshape(-1)
            # bbox_conf = np.transpose(sigmoid(outputs[3*i+1][idx]),(1, 2, 0)).reshape(-1, 1)
            bbox_reg = outputs[i+5][idx].reshape(-1, 4)
            anchor = anchors[i]

            bboxs, classs, score = decode(bbox_reg, bbox_cls, anchor, int(width)//outputs_size_list[i][-1] ,outputs_size_list[i])
            

            if bboxs.size != 0:
                bboxs /= (int(width), int(height), int(width), int(height))
                # bboxs /= (int(height), int(width), int(height), int(width))
                bboxes.append(bboxs)
                classes.append(classs)
                scores.append(score)

        if len(bboxes) != 0:
            bboxes = np.concatenate(bboxes)
            classes = np.concatenate(classes)
            scores = np.concatenate(scores)
        else:
            bboxes = np.array(bboxes)
            classes = np.array(classes)
            scores = np.array(scores)

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


def get_retina_bboxes_v1(outputs, batch_size, params):

    channel, height, width = params["input_size"].split(",")

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs_size_list = np.array(outputs_size_list)[[5,6,7,8,9,0,1,2,3,4]]
    # outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]
    # anchors = Anchors((int(height), int(width)))
    strides = [8,16,32,64,128]
    anchors = generate_anchors_v1(strides, [0.5, 1.0, 2.0],[4 * 2 ** (i / 3) for i in range(3)])

    npreds = []
    for idx in range(batch_size):
        bboxes, priors, scores = [], [] ,[]
        for i in range(len(outputs)//2):       
            # bbox_cls = sigmoid(outputs[i][idx].reshape(-1))     
            # bbox_reg = outputs[i+5][idx].reshape(-1, 4)
            bbox_cls = sigmoid(outputs[i][idx].transpose(1,2,0).reshape(-1,80))
            bbox_reg = outputs[i+5][idx].transpose(1,2,0).reshape(-1, 4)
            anchor = anchors[i]
            xv, yv = np.meshgrid(np.arange(outputs_size_list[i][-2]), np.arange(outputs_size_list[i][-1]))
            xv, yv = xv.reshape(-1)*strides[i], yv.reshape(-1)*strides[i]
            shifts = np.stack([xv,yv,xv,yv], axis=1)
            new_anchor = (anchor[None,:,:] + shifts[:,None,:]).reshape(-1,4)

            bboxs, score, prior = decode_v1(bbox_reg, bbox_cls, new_anchor, strides[i] ,outputs_size_list[i])
            

            if bboxs.size != 0:
                # boxes /= (int(width), int(height), int(width), int(height))
                # boxes /= (int(height), int(width), int(height), int(width))
                bboxes.append(bboxs)
                priors.append(prior)
                scores.append(score)

        if len(bboxes) != 0:
            bboxes = np.concatenate(bboxes)
            priors = np.concatenate(priors)
            scores = np.concatenate(scores)
        else:
            bboxes = np.array(bboxes)
            priors = np.array(priors)
            scores = np.array(scores)

        bboxes = delta2box(bboxes, priors, None, None)
        classes = scores.argmax(axis=-1)
        scores = scores.max(axis=-1)
        
        if bboxes.size != 0:
            bboxes /= (int(width), int(height), int(width), int(height))
            keep = np.where(scores >= 0.05)
            bboxes = bboxes[keep]
            classes = classes[keep]
            scores = scores[keep]
            
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


def get_retina_pad_bboxes(outputs, batch_size, params):

    channel, height, width = params["input_size"].split(",")

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs_size_list = np.array(outputs_size_list)[[5,6,7,8,9,0,1,2,3,4]]
    # outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]
    # anchors = Anchors((int(height), int(width)))
    strides = [8,16,32,64,128]
    anchors = generate_anchors_pd(strides, [0.5, 1.0, 2.0],[4 * 2 ** (i / 3) for i in range(3)])
    anchors = [np.array([anchor[0], anchor[3],anchor[6],anchor[1],anchor[4],anchor[7],anchor[2],anchor[5],anchor[8]]) for anchor in anchors]

    npreds = []
    for idx in range(batch_size):
        bboxes, priors, scores = [], [] ,[]
        for i in range(len(outputs)//2):               
            bbox_cls = sigmoid(outputs[i][idx].transpose(1,2,0).reshape(-1,80))
            bbox_reg = outputs[i+5][idx].transpose(1,2,0).reshape(-1, 4)
            anchor = anchors[i]
            xv, yv = np.meshgrid(np.arange(outputs_size_list[i][-2]), np.arange(outputs_size_list[i][-1]))
            xv, yv = xv.reshape(-1)*strides[i], yv.reshape(-1)*strides[i]
            shifts = np.stack([xv,yv,xv,yv], axis=1)
            new_anchor = (anchor[None,:,:] + shifts[:,None,:]).reshape(-1,4)

            bboxs, score, prior = decode_v1(bbox_reg, bbox_cls, new_anchor, strides[i] ,outputs_size_list[i])
            

            if bboxs.size != 0:
                # boxes /= (int(width), int(height), int(width), int(height))
                # boxes /= (int(height), int(width), int(height), int(width))
                bboxes.append(bboxs)
                priors.append(prior)
                scores.append(score)

        if len(bboxes) != 0:
            bboxes = np.concatenate(bboxes)
            priors = np.concatenate(priors)
            scores = np.concatenate(scores)
        else:
            bboxes = np.array(bboxes)
            priors = np.array(priors)
            scores = np.array(scores)

        bboxes = delta2box(bboxes, priors, None, None)
        classes = scores.argmax(axis=-1)
        scores = scores.max(axis=-1)
        
        if bboxes.size != 0:
            bboxes /= (int(width), int(height), int(width), int(height))
            keep = np.where(scores >= 0.05)
            bboxes = bboxes[keep]
            classes = classes[keep]
            scores = scores[keep]
            
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