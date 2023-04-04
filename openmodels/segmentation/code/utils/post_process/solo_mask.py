import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_seg(cate_preds, kernel_preds, seg_pred, img_shape, ori_shape):
    num_levels = len(cate_preds)
    featmap_size = seg_pred.size()[-2:]

    result_list = []
    for img_id in range(1):
        cate_pred_list = [
            cate_preds[i][img_id].view(-1, 80).detach() for i in range(num_levels)
        ]
        seg_pred_list = seg_pred[img_id, ...].unsqueeze(0)
        kernel_pred_list = [
            kernel_preds[i][img_id].permute(1, 2, 0).view(-1, 256).detach()
                            for i in range(num_levels)
        ]
        # img_shape = img_metas[img_id]['img_shape']
        # ori_shape = img_metas[img_id]['ori_shape']

        cate_pred_list = torch.cat(cate_pred_list, dim=0)
        kernel_pred_list = torch.cat(kernel_pred_list, dim=0)
        # cate_pred_list = np.concatenate(cate_pred_list, axis=0)
        # kernel_pred_list = np.concatenate(kernel_pred_list, axis=0)

        result = get_seg_single(cate_pred_list, seg_pred_list, kernel_pred_list,
                                    featmap_size, img_shape, ori_shape)
        result_list.append(result)
    return result_list

def get_seg_single(cate_preds,
                    seg_preds,
                    kernel_preds,
                    featmap_size,
                    img_shape,
                    ori_shape):

    assert len(cate_preds) == len(kernel_preds)
    seg_num_grids = [40,36,24,16,12]
    _strides = [8,8,16,32,32]

    # overall info.
    h, w = img_shape[:2]
    upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

    # process.
    inds = (cate_preds > 0.1)
    cate_scores = cate_preds[inds]
    if len(cate_scores) == 0:
        return None

    # cate_labels & kernel_preds
    inds = inds.nonzero()
    cate_labels = inds[:, 1]
    kernel_preds = kernel_preds[inds[:, 0]]

    # trans vector.
    size_trans = cate_labels.new_tensor(seg_num_grids).pow(2).cumsum(0)
    strides = kernel_preds.new_ones(size_trans[-1])

    n_stage = len(seg_num_grids)
    strides[:size_trans[0]] *= _strides[0]
    for ind_ in range(1, n_stage):
        strides[size_trans[ind_-1]:size_trans[ind_]] *= _strides[ind_]
    strides = strides[inds[:, 0]]

    # mask encoding.
    I, N = kernel_preds.shape
    kernel_preds = kernel_preds.view(I, N, 1, 1)
    seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()
    # mask.
    seg_masks = seg_preds > 0.5
    sum_masks = seg_masks.sum((1, 2)).float()

    # filter.
    keep = sum_masks > strides
    if keep.sum() == 0:
        return None

    seg_masks = seg_masks[keep, ...]
    seg_preds = seg_preds[keep, ...]
    sum_masks = sum_masks[keep]
    cate_scores = cate_scores[keep]
    cate_labels = cate_labels[keep]

    # maskness.
    seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
    cate_scores *= seg_scores

    # sort and keep top nms_pre
    sort_inds = torch.argsort(cate_scores, descending=True)
    if len(sort_inds) > 500:
        sort_inds = sort_inds[:500]
    seg_masks = seg_masks[sort_inds, :, :]
    seg_preds = seg_preds[sort_inds, :, :]
    sum_masks = sum_masks[sort_inds]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    # Matrix NMS
    cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                kernel='gaussian',sigma=2.0, sum_masks=sum_masks)

    # filter.
    keep = cate_scores >= 0.05
    if keep.sum() == 0:
        return None
    seg_preds = seg_preds[keep, :, :]
    cate_scores = cate_scores[keep]
    cate_labels = cate_labels[keep]

    # sort and keep top_k
    sort_inds = torch.argsort(cate_scores, descending=True)
    if len(sort_inds) > 100:
        sort_inds = sort_inds[:100]
    seg_preds = seg_preds[sort_inds, :, :]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                size=upsampled_size_out,
                                mode='bilinear')[:, :, :h, :w]
    seg_masks = F.interpolate(seg_preds,
                            size=ori_shape[:2],
                            mode='bilinear').squeeze(0)
    seg_masks = seg_masks > 0.5
    return seg_masks, cate_labels, cate_scores

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep

def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss' 
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay 
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update

def get_solo_mask(outputs, batch_size, params,content):
    channel, height, width = params["input_size"].split(",")

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].copy().reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]

    npreds = []
    for idx in range(batch_size):
        outputs_torch = [torch.from_numpy(output) for output in outputs]
        kernel_preds = outputs_torch[1:6]
        cate_preds = [points_nms(output_torch.sigmoid(), kernel=2).permute(0, 2, 3, 1) for output_torch in outputs_torch[6:]]
        seg_pred = outputs_torch[0]

        ori_size = [int(content[idx][3]), int(content[idx][2])]
        scale_facter = float(height)/max(ori_size)
        img_size = [int(orisize * scale_facter) for orisize in ori_size]

        result = get_seg(cate_preds, kernel_preds, seg_pred, img_size, ori_size)[0]

        npreds.append(result)

    return npreds

