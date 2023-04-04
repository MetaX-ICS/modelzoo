from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import torch.nn.functional as F
import pycocotools
import numpy as np
import json
import uuid
from PIL import Image

from .tools import preprocess_test
from .deeplabv3_mask import voc_class, COCOStuffeval

coco80_to_coco91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 
56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 
67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def sanitize_coordinates(_x1, _x2, img_size:int, padding:int=0, cast:bool=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1-padding, min=0)
    x2 = torch.clamp(x2+padding, max=img_size)

    return x1, x2

def crop(masks, boxes, padding:int=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)
    
    masks_left  = rows >= x1.view(1, 1, -1)
    masks_right = rows <  x2.view(1, 1, -1)
    masks_up    = cols >= y1.view(1, 1, -1)
    masks_down  = cols <  y2.view(1, 1, -1)
    
    crop_mask = masks_left * masks_right * masks_up * masks_down
    
    return masks * crop_mask.float()

def postprocess(det_output, w, h, interpolation_mode='bilinear', score_threshold=0):

    dets = det_output#[batch_idx]
    # dets = dets['detection']

    if dets is None:
        return [torch.Tensor()] * 4 # Warning, this is 4 copies of the same thing

    if score_threshold > 0:
        keep = dets['score'] > score_threshold

        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]
        
        if dets['score'].size(0) == 0:
            return [torch.Tensor()] * 4
    
    # Actually extract everything from dets now
    classes = dets['class']
    boxes   = dets['box']
    scores  = dets['score']
    masks   = dets['mask']

    proto_data = dets['proto']

    
    # if visualize_lincomb:
    #     display_lincomb(proto_data, masks)
    masks = proto_data @ masks.t()
    masks = torch.sigmoid(masks)
    masks = crop(masks, boxes)
    # Permute into the correct output shape [num_dets, proto_h, proto_w]
    masks = masks.permute(2, 0, 1).contiguous()

    # Scale masks up to the full image
    # masks_bak =  Image.fromarray(masks).convert("P")
    # masks_bak = np.array(masks_bak.resize((h, w), Image.BILINEAR))
    masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)

    # Binarize the masks
    masks.gt_(0.5)
    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
    boxes = boxes.long()

    return classes, scores, boxes, masks

def comute_coco_scores(predns, params, modelname, datanums):
    label_list = preprocess_test(params["image_T4"])
    
    coco_result = []
    for i in range(datanums):
        label_name, img_str, w, h = label_list[i]

        classes, scores, boxes, masks = postprocess(predns[i], int(w), int(h))
        if classes.size(0) == 0:
            continue

        classes = list(classes.cpu().numpy().astype(int))
        if isinstance(scores, list):
            box_scores = list(scores[0].cpu().numpy().astype(float))
            mask_scores = list(scores[1].cpu().numpy().astype(float))
        else:
            scores = list(scores.cpu().numpy().astype(float))
            box_scores = scores
            mask_scores = scores

        boxes = boxes.numpy()
        masks = masks.view(-1, int(h), int(w)).cpu().numpy()

        for i in range(masks.shape[0]):
            # Make sure that the bounding box actually makes sense and a mask was produced
            if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                rle = pycocotools.mask.encode(np.asfortranarray( masks[i,:,:].astype(np.uint8)))
                rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings
                coco_result.append(
                    {
                        'image_id': int(img_str),
                        'category_id': coco80_to_coco91[int(classes[i])],
                        'segmentation': rle,
                        'score': float(mask_scores[i])
                    }
                )
    
    str_uuid = str(uuid.uuid1())[:8]
    output_name = modelname + "/coco_prediction_{}.json".format(str_uuid)
    with open(output_name, 'w') as f:
        json.dump(coco_result, f)
    gt_annotations = COCO("./data/instances_val2017.json")
    mask_dets = gt_annotations.loadRes(output_name)
    imgIds=sorted(mask_dets.getImgIds())
    mask_eval = COCOeval(gt_annotations, mask_dets, 'segm')
    mask_eval.params.imgIds  = imgIds[:datanums]
    mask_eval.evaluate()
    mask_eval.accumulate()
    mask_eval.summarize()

    mAP, mAP50 = mask_eval.stats[:2]
    
    return mAP, mAP50

def comute_coco_scores_v1(predns, params, modelname, datanums):
    label_list = preprocess_test(params["image_T4"])
    
    coco_result = []
    for i in range(datanums):
        label_name, img_str, w, h = label_list[i]
        cur_result = predns[i]
        if cur_result is None:
            continue
        masks = cur_result[0].numpy().astype(np.uint8)
        classes = cur_result[1].numpy()
        mask_scores = cur_result[2].numpy()
        num_ins = masks.shape[0]

        for i in range(num_ins):
            rle = pycocotools.mask.encode(np.asfortranarray( masks[i,:,:]))
            rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings
            coco_result.append(
                {
                    'image_id': int(img_str),
                    'category_id': coco80_to_coco91[int(classes[i])],
                    'segmentation': rle,
                    'score': float(mask_scores[i])
                }
            )

    str_uuid = str(uuid.uuid1())[:8]
    output_name = modelname + "/coco_prediction_{}.json".format(str_uuid)
    with open(output_name, 'w') as f:
        json.dump(coco_result, f)
    gt_annotations = COCO("./data/instances_val2017.json")
    mask_dets = gt_annotations.loadRes(output_name)
    imgIds=sorted(mask_dets.getImgIds())
    mask_eval = COCOeval(gt_annotations, mask_dets, 'segm')
    mask_eval.params.imgIds  = imgIds[:datanums]
    mask_eval.evaluate()
    mask_eval.accumulate()
    mask_eval.summarize()

    mAP, mAP50 = mask_eval.stats[:2]
    
    return mAP, mAP50

def comute_voc2coco_scores(predns, params, modelname, datanums):
    str_uuid = str(uuid.uuid1())[:8]
    output_name =modelname + "/voc2coco_prediction_{}.json".format(str_uuid)
    with open(output_name, "w") as f:
        json.dump(predns, f)

    cocoGt = COCO(params["label_list"])
    cocoRes = cocoGt.loadRes(output_name)

    cocoEval = COCOStuffeval(cocoGt, cocoRes, subclass=voc_class, isStuff=False, stuffStartId=1, stuffEndId=91)

    imgIds=sorted(cocoGt.getImgIds())
    cocoEval.params.imgIds  = imgIds[:datanums]
    cocoEval.evaluate()
    cocoEval.summarize()

    mIOU, fwIOU, mAcc, pAcc  = cocoEval.stats

    return mIOU, mAcc, pAcc
