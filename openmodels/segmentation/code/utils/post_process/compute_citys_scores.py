import time
import os
import sys

import numpy as np
from PIL import Image

import torch

from .tools import preprocess_test

_key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
_mapping = np.array(range(-1, len(_key) - 1)).astype('int32')

class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.nclass)

            self.total_correct += correct
            self.total_label += labeled
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0


# pytorch version
def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()



def get_gtmask(image_file, size):
    gt_mask = Image.open(image_file)
    short_size = size[0]
    w, h = gt_mask.size
    if w > h:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    gt_mask = gt_mask.resize((ow, oh), Image.NEAREST)
    # center crop
    w, h = gt_mask.size
    x1 = int(round((w - short_size) / 2.))
    y1 = int(round((h - short_size) / 2.))
    gt_mask = gt_mask.crop((x1, y1, x1 + short_size, y1 + short_size))
    gt_mask = np.array(gt_mask).astype('int32')

    values = np.unique(gt_mask)
    for value in values:
        assert (value in _mapping)
    index = np.digitize(gt_mask.ravel(), _mapping, right=True)
    gt_mask = _key[index].reshape(gt_mask.shape)

    gt_mask = torch.LongTensor(np.array(gt_mask).astype('int32'))
    
    return gt_mask.reshape((1,-1,*size))

    

def comute_citys_scores(predns, params, modelname, datanums):
    label_list = preprocess_test(params["image_T4"])
    ann_dir="./data/Cityscapes_data/gtFine/val/"
    metric = SegmentationMetric(19)
    # result_file_path="/opt/yrong/Dataset/data/enet_result/fp32/"
    # file_template="iter_%s_attach_ConvTranspose_ConvTranspose_341_out0_0_out0_1_19_480_480.tensor"

    # pred_masks = []
    # gt_masks = []
    for i in range(datanums):
        label_name, _, H, W = label_list[i]
        mask = predns[i]
        c, netH, netW = predns[i].shape
        mask = mask.reshape([1, c, netH, netW])
        # mask_=np.loadtxt(result_file_path+file_template%(i), dtype=np.float32)
        # mask_=np.reshape(mask_,[1,19,480,480])
        

        label_name = label_name.split("/")
        label = ann_dir + label_name[-2]+"/" + label_name[-1].replace("leftImg8bit","gtFine_labelIds")
        gt_mask = get_gtmask(label,(netH, netW))

        preds = torch.from_numpy(mask)

        metric.update(preds, gt_mask)
        pixAcc, mIoU = metric.get()
        if i == datanums-1:
            print("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, pixAcc * 100, mIoU * 100))

    # eval_results = voc2012_evaluation_v1(pred_masks, gt_masks)
    return (pixAcc, mIoU)
