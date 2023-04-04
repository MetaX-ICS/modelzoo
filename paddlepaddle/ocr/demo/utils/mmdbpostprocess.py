import numpy as np
import cv2
from .db_postprocess import DBPostProcess
from shapely.geometry import Polygon
import pyclipper

def points2boundary(points, text_repr_type, text_score=None, min_width=-1):

    assert isinstance(points, np.ndarray)
    assert points.shape[1] == 2
    assert text_repr_type in ['quad', 'poly']
    assert text_score is None or 0 <= text_score <= 1

    if text_repr_type == 'quad':
        rect = cv2.minAreaRect(points)
        vertices = cv2.boxPoints(rect)
        boundary = []
        if min(rect[1]) > min_width:
            boundary = [p for p in vertices.flatten().tolist()]

    elif text_repr_type == 'poly':

        height = np.max(points[:, 1]) + 10
        width = np.max(points[:, 0]) + 10

        mask = np.zeros((height, width), np.uint8)
        mask[points[:, 1], points[:, 0]] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boundary = list(contours[0].flatten().tolist())

    if text_score is not None:
        boundary = boundary + [text_score]
    if len(boundary) < 8:
        return None

    return boundary

def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded

def valid_boundary(x, with_score=True):
    num = len(x)
    if num < 8:
        return False
    if num % 2 == 0 and (not with_score):
        return True
    if num % 2 == 1 and with_score:
        return True

    return False

def dbnet_postprocess(preds,scale_factor):
    # print("111111:",preds.shape)
    preds = preds.squeeze()
    mask_thr = 0.3
    max_candidates = 3000
    epsilon_ratio = 0.01
    min_text_score = 0.3
    unclip_ratio = 1.5
    # text_repr_type = "poly"
    text_repr_type = "quad"
    min_text_width = 5
    prob_map = preds[0, :, :]
    text_mask = prob_map > mask_thr
    score_map = prob_map.astype(np.float32)
    text_mask = text_mask.astype(np.uint8)  # to numpy
    contours, _ = cv2.findContours((text_mask * 255).astype(np.uint8),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boundaries = []
    for i, poly in enumerate(contours):
        if i > max_candidates:
            break
        epsilon = epsilon_ratio * cv2.arcLength(poly, True)
        approx = cv2.approxPolyDP(poly, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(score_map, points)
        if score < min_text_score:
            continue
        poly = unclip(points, unclip_ratio=unclip_ratio)
        if len(poly) == 0 or isinstance(poly[0], list):
            continue
        poly = poly.reshape(-1, 2)
        # print(poly.shape)
        if text_repr_type == 'quad':
            poly = points2boundary(poly, text_repr_type, score, min_text_width)
        elif text_repr_type == 'poly':
            poly = poly.flatten().tolist()
            if score is not None:
                poly = poly + [score]
            if len(poly) < 8:
                poly = None
        # print(len(poly))

        if poly is not None:
            boundaries.append(poly)
    rescale = True
    if rescale:
        for b in boundaries:
            sz = len(b)
            valid_boundary(b, True)
            b[:sz -1] = (np.array(b[:sz - 1]) /
                        (np.tile(scale_factor[:2], int((sz - 1) / 2)).reshape(1, sz - 1))).flatten().tolist()
    return boundaries



def get_mmdbnet_post(outputs, batch_size, params, content):

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    outputs = [outputs[i].reshape(-1,outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]
    
    db_processs = DBPostProcess(thresh=0.3,
                                box_thresh=0.6,
                                max_candidates=1000,
                                unclip_ratio=1.5)
    preds = []

    for idx in range(batch_size):
        output = outputs[0][idx].copy()

        # out_preds = np.expand_dims(output, axis=0)
        # bboxs = dbnet_postprocess(out_preds,np.array([736/720.,1312/1280.,736/720.,1312/1280.]))
        # bboxes = [np.array(box[:8]).astype(np.int16) for box in bboxs]
        pred_dict = {}
        pred_dict["maps"] = np.expand_dims(output, axis=0)
        bboxs_1 = db_processs(pred_dict, np.array([[720.,1280.,736/720.,1312/1280.]]))
        bboxes_1 = [box.reshape(-1) for box in bboxs_1[0]]


        preds.append(bboxes_1)
    return preds