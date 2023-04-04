import numpy as np
import collections
from itertools import product

from .nms import filter_bbox, nms_bboxes, sigmoid, softmax


def Anchors(input_data_size):

    image_shape = input_data_size

    pyramid_levels = [3, 4, 5, 6, 7]
    strides = [2 ** x for x in pyramid_levels]
    scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scale = 4.
    # aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    # num_scales = len([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    # num_anchors = len(aspect_ratios) * num_scales
    boxes_all = []
    for stride in strides:
        boxes_level = []
        for scale, ratio in product(scales, ratios):
            if image_shape[1] % stride != 0:
                raise ValueError('input size must be divided by the stride.')
            base_anchor_size = anchor_scale * stride * scale
            anchor_size_x_half = base_anchor_size * ratio[0] / 2.0
            anchor_size_y_half = base_anchor_size * ratio[1] / 2.0

            x = np.arange(stride / 2, image_shape[1], stride)
            y = np.arange(stride / 2, image_shape[0], stride)
            xv, yv = np.meshgrid(x, y)
            xv = xv.reshape(-1)
            yv = yv.reshape(-1)

            # y1,x1,y2,x2
            boxes = np.vstack((yv - anchor_size_y_half, xv - anchor_size_x_half,
                                yv + anchor_size_y_half, xv + anchor_size_x_half))
            boxes = np.swapaxes(boxes, 0, 1)
            boxes = np.expand_dims(boxes, axis=1) 
            boxes_level.append(boxes)
            
        # concat anchors on the same level to the reshape NxAx4
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))
    
    anchor_boxes = np.vstack(boxes_all)
    # anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)
    # anchor_boxes = anchor_boxes.unsqueeze(0)
    anchor_boxes = np.expand_dims(anchor_boxes,0)

    # save it for later use to reduce overhead
    # self.last_anchors[image.device] = anchor_boxes
    return anchor_boxes

def BBoxTransform(anchors, regression):

    y_ca = (anchors[..., 0] + anchors[..., 2]) / 2
    x_ca = (anchors[..., 1] + anchors[..., 3]) / 2
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]

    w = np.exp(regression[..., 3]) * wa
    h = np.exp(regression[..., 2]) * ha

    y_centers = regression[..., 0] * ha + y_ca
    x_centers = regression[..., 1] * wa + x_ca

    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    # ymax = y_centers + h / 2.
    # xmax = x_centers + w / 2.

    return np.stack((xmin, ymin, w, h), axis=-1)

def get_effc_bboxes(outputs, batch_size, params):

    channel, height, width = params["input_size"].split(",")

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1]) for i in range(len(outputs))]
    anchors = Anchors((int(height), int(width)))

    npreds = []
    for idx in range(batch_size):
        # bboxes, classes, scores = [], [] ,[]

        bbox = outputs[0][idx]
        bbox_cls = outputs[1][idx]

        bbox = BBoxTransform(anchors, bbox).squeeze()
        bbox /= (int(width), int(height), int(width), int(height))

        bboxes, classes, scores = filter_bbox(bbox, bbox_cls, 1., 0.05)
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
