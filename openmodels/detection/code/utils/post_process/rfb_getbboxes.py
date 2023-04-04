from math import sqrt
from itertools import product
import numpy as np

from .nms import nms_bboxes, filter_bbox, sigmoid, softmax

num_classes = 81

def PriorBox(clip=True):
    feature_maps = [64, 32, 16, 8, 4, 2, 1]
    image_size = 512
    steps = [8, 16, 32, 64, 128, 256, 512]
    min_sizes = [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8]
    max_sizes = [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72]
    aspect_ratios = [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]]

    mean = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f), repeat=2):
            f_k = image_size / steps[k]
            cx = (j + 0.5) / f_k
            cy = (i + 0.5) / f_k

            s_k = min_sizes[k]/image_size
            mean += [cx, cy, s_k, s_k]

            # aspect_ratio: 1
            # rel size: sqrt(s_k * s_(k+1))
            s_k_prime = sqrt(s_k * (max_sizes[k]/image_size))
            mean += [cx, cy, s_k_prime, s_k_prime]

            # rest of aspect ratios
            for ar in aspect_ratios[k]:
                mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

    # back to torch land
    # output = torch.Tensor(mean).view(-1, 4)
    output = np.array(mean).reshape(-1, 4)
    if clip:
        output.clip(max=1, min=0)
    return output

def decode(loc, priors, variances):
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    # boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_bbox(loc, conf, priors):
    # loc, conf = predictions

    num = 1 #loc.shape[0]  # batch size
    num_priors = priors.shape[0]
    boxes = np.zeros((1, num_priors, 4), np.float32)
    scores = np.zeros((1, num_priors, num_classes), np.float32)

    # if num == 1:
    #     # size batch x num_classes x num_priors
    #     # conf_preds = conf.unsqueeze(0)
    #     conf_preds = np.expand_dims(conf, axis=0)

    # else:
    #     conf_preds = conf.reshape(num, num_priors,
    #                                 num_classes)
    #     boxes.expand_(num, num_priors, 4)
    #     scores.expand_(num, num_priors, num_classes)

    # Decode predictions into bboxes.
    # for i in range(num):
    decoded_boxes = decode(loc, priors, [0.1,0.2])
    conf_scores = conf#_preds[i]#.clone()

        # boxes[i] = decoded_boxes
        # scores[i] = conf_scores

    return decoded_boxes, conf_scores

def get_rfb_bboxes(outputs, batch_size, params):
    channel, height, width = params["input_size"].split(",")
    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1]) for i in range(len(outputs))]
    priors = PriorBox()

    npreds = []
    for idx in range(batch_size):
        bboxes, classes, scores = [], [] ,[]

        boxes, scores = decode_bbox(outputs[0][idx], outputs[1][idx], priors)

        bboxes, classes, scores = filter_bbox(boxes, scores[:,1:], 1.)
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
            preds = np.concatenate((nbboxes, np.expand_dims(nclasses, axis=-1), np.expand_dims(nscores, axis=-1)), axis=-1)
            npreds.append(preds)
        else:
            npreds.append(np.array([]))

    return npreds

