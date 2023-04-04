import time
import os
import sys

import numpy as np
from PIL import Image

from terminaltables import AsciiTable

from tensorflow.keras.utils import to_categorical
from .tools import preprocess_test



CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor')


def dice_coeff(inputs, targets, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert inputs.shape == targets.shape
    if len(inputs.shape) == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {inputs.shape})')

    if len(inputs.shape) == 2 or reduce_batch_first:
        inter = np.dot(inputs.reshape(-1), targets.reshape(-1))
        sets_sum = np.sum(inputs) + np.sum(targets)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(inputs.shape[0]):
            dice += dice_coeff(inputs[i, ...], targets[i, ...])
        return dice / inputs.shape[0]

def multiclass_dice_coeff(inputs, targets, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert inputs.shape == targets.shape
    dice = 0
    for channel in range(inputs.shape[0]):
        dice += dice_coeff(inputs[channel, ...], targets[channel, ...], reduce_batch_first, epsilon)

    return dice / inputs.shape[0]

def comute_unet_scores(predns, params, modelname, datanums):
    label_list = preprocess_test(params["image_T4"])
    dice_score = 0

    for i in range(datanums):
        _, newH, newW = predns[i].shape
        # mask_true = cv2.imread("./data/" + label_list[i][0])
        label = "./data/masks/{}_mask.gif".format(label_list[i][0].split('/')[-1][:-4])
        mask_true = Image.open(label)
        mask_true = mask_true.resize((newW, newH))
        mask_true = np.asarray(mask_true)

        mask_true = np.expand_dims(mask_true, axis=-1)   
        mask_true = np.transpose(to_categorical(mask_true,2), (2, 0, 1))

        dice_score += multiclass_dice_coeff(predns[i][:1, ...], mask_true[:1, ...], reduce_batch_first=False)

    return dice_score / datanums

def postprocess_mask(mask, image_size, net_input_width, net_input_height):
    h = image_size[0]
    w = image_size[1]
    scale = min(net_input_width / w, net_input_height / h)

    # pad_w = net_input_width - w * scale
    # pad_h = net_input_height - h * scale
    # pad_left = (pad_w // 2)
    # pad_top = (pad_h // 2)
    # if pad_top < 0:
    #     pad_top = 0
    # if pad_left < 0:
    #     pad_left = 0
    # pad_left = int(pad_left)
    # pad_top = int(pad_top)
    # a = int(500 - pad_top)
    # b = int(500 - pad_left)
    # mask = mask[pad_top:a, pad_left:b]
    mask = mask[:int(h * scale), :int(w * scale)]
    mask = np.array(Image.fromarray(mask.astype(np.float32)).resize((int(w), int(h)),Image.BILINEAR)).astype(np.int32)
    return mask

def voc2012_evaluation_v1(results, gt_seg_maps):
    metric = ['mIoU']
    eval_results = {}

    num_classes = len(CLASSES)
    ignore_index = 255
    label_map = dict()
    reduce_zero_label = False

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes,))
    total_area_union = np.zeros((num_classes,))
    total_area_pred_label = np.zeros((num_classes,))
    total_area_label = np.zeros((num_classes,))
    for i in range(num_imgs):
        pred_label = results[i]
        label = gt_seg_maps[i]

        if label_map is not None:
            for old_id, new_id in label_map.items():
                label[label == old_id] = new_id
        if reduce_zero_label:
            label[label == 0] = 255
            label = label - 1
            label[label == 254] = 255

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = np.histogram(
            intersect.astype(np.float32), bins=np.arange(num_classes+1))
        area_pred_label = np.histogram(
            pred_label.astype(np.float32), bins=np.arange(num_classes+1))
        area_label = np.histogram(
            label.astype(np.float32), bins=np.arange(num_classes+1))
        area_union = area_pred_label[0] + area_label[0] - area_intersect[0]

        total_area_intersect += area_intersect[0]
        total_area_union += area_union
        total_area_pred_label += area_pred_label[0]
        total_area_label += area_label[0]
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    ret_metrics = [all_acc, acc]
    iou = total_area_intersect / total_area_union
    ret_metrics.append(iou)
    # ret_metrics = [metric.numpy() for metric in ret_metrics]

    class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
    class_names = CLASSES

    ret_metrics_round = [
        np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
    ]
    for i in range(num_classes):
        class_table_data.append([class_names[i]] +
                                [m[i] for m in ret_metrics_round[2:]] +
                                [ret_metrics_round[1][i]])
    summary_table_data = [['Scope'] +
                          ['m' + head
                           for head in class_table_data[0][1:]] + ['aAcc']]
    ret_metrics_mean = [
        np.round(np.nanmean(ret_metric) * 100, 2)
        for ret_metric in ret_metrics
    ]
    summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                              [ret_metrics_mean[1]] +
                              [ret_metrics_mean[0]])

    print('per class results:')
    table = AsciiTable(class_table_data)
    print('\n' + table.table)
    print('Summary:')
    table = AsciiTable(summary_table_data)
    print('\n' + table.table)

    for i in range(1, len(summary_table_data[0])):
        eval_results[summary_table_data[0]
        [i]] = summary_table_data[1][i] / 100.0
    for idx, sub_metric in enumerate(class_table_data[0][1:], 1):
        for item in class_table_data[1:]:
            eval_results[str(sub_metric) + '.' +
                         str(item[0])] = item[idx] / 100.0
    return eval_results

def comute_voc_scores(predns, params, modelname, datanums):
    label_list = preprocess_test(params["image_T4"])
    ann_dir="./data/voc2012_data/VOCdevkit/VOC2012/SegmentationClass/"

    pred_masks = []
    gt_masks = []
    for i in range(datanums):
        _, label, H, W = label_list[i]
        mask = predns[i]
        netH, netW = predns[i].shape
        mask = postprocess_mask(mask, (int(W),int(H)), netW, netH)

        label = ann_dir + label + ".png"
        gt_mask = np.array(Image.open(label))

        pred_masks.append(mask)
        gt_masks.append(gt_mask)

    eval_results = voc2012_evaluation_v1(pred_masks, gt_masks)
    return eval_results






