import numpy as np
import collections
from itertools import product

from .nms import filter_bbox, sigmoid, softmax


def nms_bboxes(boxes, scores, isxyxy=False):
    if not isxyxy:
        boxes[:, 0] = boxes[:,0]
        boxes[:, 1] = boxes[:,1]
        boxes[:, 2] = (boxes[:,2]+boxes[:,0]) - boxes[:,0]
        boxes[:, 3] = (boxes[:,3]+boxes[:,1]) - boxes[:,1]

        x = boxes[:,0]
        y = boxes[:,1]
        w = boxes[:,2]
        h = boxes[:,3]
    else:
        x = boxes[:,0]
        y = boxes[:,1]
        w = boxes[:,2] - x
        h = boxes[:,3] - y

    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i]+w[i], x[order[1:]]+w[order[1:]])
        yy2 = np.minimum(y[i]+h[i], y[order[1:]]+h[order[1:]])

        w1 = np.maximum(.0, xx2 - xx1 + 0.000001)
        h1 = np.maximum(.0, yy2 - yy1 + 0.000001)
        inter = w1 * h1

        iou = inter / (areas[i] + areas[order[1:]] - inter + 0.000001)
        inds = np.where(iou <= 0.45)[0]
        order = order[inds+1]
    
    keep = np.array(keep)
    return keep


def create_ssd_anchors(num_layers=6, min_scale=0.2,
                       max_scale=0.95,
                       scales=None,
                       aspect_ratios=(1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3),
                    #    aspect_ratios=(1.0, 2.0, 1.0 / 2, 3.0, 1.0 / 3),
                       interpolated_scale_aspect_ratio=1.0,
                    #    base_anchor_size=None,
                    #    anchor_strides=None,
                    #    anchor_offsets=None,
                       reduce_boxes_in_lowest_layer=True):
 
    #base_anchor_size = [1.0,1,0]
    # base_anchor_size = tf.constant(base_anchor_size, dtype=tf.float32)
    box_specs_list = []
    
    #min_scale = 0.2,  max_scale = 0.95, num_layers = 6, 这三个参数在"ssd_mobilenet_v1_pets.config"中定义
    #scales = [0.2, 0.35 , 0.5, 0.65 0.80, 0.95, 1]
    scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
                for i in range(num_layers)] + [1.0]
    
    for layer, scale, scale_next in zip(
        range(num_layers), scales[:-1], scales[1:]):
        layer_box_specs = []
        if layer == 0 and reduce_boxes_in_lowest_layer:
        #layer =0时，只有三组box, 如何选取？？
            layer_box_specs = [(0.1, 1.0), (scale, 2.0), (scale, 0.5)]
        else:  
            #aspect_ratios= [1.0, 2.0 , 0.5, 3.0, 0.3333]
            #aspect_ratios可在"ssd_mobilenet_v1_pets.config"中自行定义
            for aspect_ratio in aspect_ratios:
                layer_box_specs.append((scale, aspect_ratio))
 
            #多增加一个anchor, aspect ratio=1, scale是现在的scale与下一层scale的乘积开平方。
            if interpolated_scale_aspect_ratio > 0.0:
                layer_box_specs.append((np.sqrt(scale*scale_next),
                                        interpolated_scale_aspect_ratio))
    
            
        box_specs_list.append(layer_box_specs)
    
    return box_specs_list

def generate_prior_bbox(box_specs_list, feature_maps = [19, 10, 5, 3, 2, 1], clip=True):#, strides = [16, 32, 64, 100, 150, 300]
    priors = []
    anchor_strides = [1.0 / feature for feature in feature_maps]
    # anchor_offsets = [0.5 * stride for stride in anchor_strides]

    for k, f in enumerate(feature_maps):
        for i, j in product(range(f), repeat=2):
            for scale, ratio in box_specs_list[k]:
                ratio = np.sqrt(ratio)
                # unit center x,y
                # cx = (j + 0.5) / (300/strides[k])
                # cy = (i + 0.5) / (300/strides[k])
                cx = (j + 0.5) * anchor_strides[k]
                cy = (i + 0.5) * anchor_strides[k]
                h = w = scale
                priors.append([cx, cy, w * ratio, h / ratio])
                # priors.append([cx, cy, w / ratio, h * ratio])

    priors = np.array(priors)
    priors = priors.clip(max=1, min=0)
    return priors

def convert_locations_to_boxes(pri_loc, proir_bbox, variances= [0.1, 0.2]):

    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    # ycenter_a, xcenter_a, ha, wa = proir_bbox[:, 1],proir_bbox[:, 0],proir_bbox[:, 3],proir_bbox[:, 2]
    bbox_x = pri_loc[...,1:2] * variances[0] * proir_bbox[..., 2:3] + proir_bbox[..., 0:1]
    bbox_y = pri_loc[...,0:1] * variances[0] * proir_bbox[..., 3:4] + proir_bbox[..., 1:2]
    bbox_w = np.exp(pri_loc[...,3:4] * variances[1]) * proir_bbox[..., 2:3]
    bbox_h = np.exp(pri_loc[...,2:3] * variances[1]) * proir_bbox[..., 3:4]
    # bbox_x = pri_loc[...,1:2] * variances[0] * proir_bbox[..., 3:4] + proir_bbox[..., 1:2]
    # bbox_y = pri_loc[...,0:1] * variances[0] * proir_bbox[..., 2:3] + proir_bbox[..., 0:1]
    # bbox_w = np.exp(pri_loc[...,3:4] * variances[1]) * proir_bbox[..., 3:4]
    # bbox_h = np.exp(pri_loc[...,2:3] * variances[1]) * proir_bbox[..., 2:3]
    bbox_x -= (bbox_w / 2.)
    bbox_y -= (bbox_h / 2.)


    return np.concatenate((bbox_x, bbox_y, bbox_w, bbox_h), axis=-1)


def get_mobilenetv1_ssd_bboxes_v1(outputs, batch_size, params):
    channel, height, width = params["input_size"].split(",")
    outputs.reverse()
    # anchors_list = np.array([float(anch) for anch in params['anchors'].split(",")]).reshape(-1,3,2)

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1]) for i in range(len(outputs))]  #,outputs_size_list[i][2]
    box_specs_list = create_ssd_anchors()
    prior_bbox = generate_prior_bbox(box_specs_list)
    
    npreds = []
    for idx in range(batch_size):
        bboxes, classes, scores = [], [] ,[]
        # for i in range(len(outputs)):

        bboxes = outputs[1][idx]
        bboxes = convert_locations_to_boxes(bboxes, prior_bbox)
        classes = outputs[0][idx]
        classes = sigmoid(classes)

        bboxes, classes, scores = filter_bbox(bboxes, classes[:,1:], 1.)
        classes += 1

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




def get_vgg16_ssd_bboxes_v1(outputs, batch_size, params, start_idx):
    # print(params)
    # channel, height, width = params["input_size"].split(",")
    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1]) for i in range(len(outputs))]  #,outputs_size_list[i][2]

    npreds = []
    # prob_threshold=0.4
    # picked_box_probs = []
    # picked_labels = []
    for idx in range(batch_size):
        bboxes = outputs[1][idx]
        # bboxes = convert_locations_to_boxes(bboxes, prior_bbox)
        classes = outputs[0][idx]

        bboxes, classes, scores = filter_bbox(bboxes, classes[:,1:], 1.)
        classes += 1

        nbboxes, nclasses, nscores = [], [], []
        for cl in set(classes):
            inds = np.where(classes == cl)
            b = bboxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = nms_bboxes(b, s, True)

            nbboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if len(nclasses) != 0 and len(nscores) != 0:    
            nbboxes = np.concatenate(nbboxes)
            nclasses = np.concatenate(nclasses)
            nscores = np.concatenate(nscores)
            indexes = np.ones((len(nbboxes), 1), dtype=np.float32) * (start_idx+idx)
            # preds = np.concatenate((nbboxes, np.expand_dims(nclasses, axis=-1), np.expand_dims(nscores, axis=-1)), axis=-1)
            preds = np.concatenate((indexes, np.expand_dims(nclasses, axis=-1), np.expand_dims(nscores, axis=-1), nbboxes), axis=-1)
            npreds.append(preds)
        # else:
            # npreds.append(np.array([]))


    return npreds


def get_resnet34_ssd_bboxes(outputs, batch_size, params):
    # print(params)
    # channel, height, width = params["input_size"].split(",")
    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1]) for i in range(len(outputs))]  #,outputs_size_list[i][2]

    npreds = []
    # prob_threshold=0.4
    # picked_box_probs = []
    # picked_labels = []
    for idx in range(batch_size):
        bboxes = outputs[0][idx]
        bboxes[:,2:4] = bboxes[:,2:4] - bboxes[:,:2]
        bbox_x = bboxes[:,1:2]
        bbox_y = bboxes[:,0:1]
        bbox_w = bboxes[:,3:4]
        bbox_h = bboxes[:,2:3]
        bboxes = np.concatenate((bbox_x, bbox_y, bbox_w, bbox_h), axis=-1)

        classes = outputs[1][idx]

        bboxes, classes, scores = filter_bbox(bboxes, classes[:,1:], 1.)
        # classes += 1

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
            # indexes = np.ones((len(nbboxes), 1), dtype=np.float32) * (start_idx+idx)
            # preds = np.concatenate((nbboxes, np.expand_dims(nclasses, axis=-1), np.expand_dims(nscores, axis=-1)), axis=-1)
            preds = np.concatenate((nbboxes, np.expand_dims(nclasses, axis=-1), np.expand_dims(nscores, axis=-1)), axis=-1)
            npreds.append(preds)
        # else:
            # npreds.append(np.array([]))

    return npreds



