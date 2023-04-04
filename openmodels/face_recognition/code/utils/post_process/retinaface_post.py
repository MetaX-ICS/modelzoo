import os
import numpy as np
from itertools import product
import uuid

from math import ceil
import tqdm
from scipy.io import loadmat

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

def nms_bboxes(boxes, scores):

    x = boxes[:,0]
    y = boxes[:,1]
    w = boxes[:,2] - boxes[:,0]
    h = boxes[:,3] - boxes[:,1]
    
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

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= 0.4)[0]
        order = order[inds+1]
    
    keep = np.array(keep)
    return keep

def filter_bbox(bboxes, scores):

    pos = np.where(scores >= 0.02)
    bboxes = bboxes[pos]  
    scores = scores[pos]

    return bboxes, scores

def generate_prior_bbox(image_size, min_sizes_list=[[16, 32], [64, 128], [256, 512]], steps=[8,16,32]):
    anchors = []
    feature_maps = [[ceil(image_size[0]/step), ceil(image_size[1]/step)] for step in steps]
    for k, f in enumerate(feature_maps):
        min_sizes = min_sizes_list[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky] 

    # back to torch land
    output = np.array(anchors).reshape(-1, 4)
    output = output.clip(max=1, min=0)
    return output

def decode(loc, priors, variances=[0.1,0.2]):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes_xy = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    boxes_wh = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    # boxes = torch.cat((
    #     priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
    #     priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes = np.concatenate((boxes_xy, boxes_wh), axis = -1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    boxes = boxes.clip(max=1, min=0)
    return boxes

def get_retinaface_post(outputs, batch_size, params):

    channel, height, width = params["input_size"].split(",")
    outputs = [outputs[0], outputs[2], outputs[1]]

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1]) for i in range(len(outputs))]

    priorbox = generate_prior_bbox(image_size=(int(height), int(width)))

    npreds = []
    for idx in range(batch_size):
        bboxes, scores = [], [] 

        bboxes = outputs[0][idx]
        scores = outputs[2][idx][...,1]

        bboxes = decode(bboxes, priorbox)


        bboxes, scores = filter_bbox(bboxes, scores)



        keep = nms_bboxes(bboxes, scores)
        if len(keep) != 0:
            bboxes = bboxes[keep]
            scores = scores[keep]

        if len(scores) != 0:
            preds = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=-1)
            npreds.append(preds)
        else:
            npreds.append(np.array([]))

    return npreds

def comute_retinaface_map(preds, params, save_path, iters):
    idx = 0
    str_uuid = str(uuid.uuid1())[:8]
    with open("./data/wider_val.txt", "r") as f:
        for img_name in f.readlines()[:iters]:
            img_info = img_name.strip().split(" ")
            img_name, width, height = img_info
            save_name = (save_path + "/" + img_name[:-4] + ".txt").replace("/images","/images_"+str_uuid)
            dirname = os.path.dirname(save_name)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            with open(save_name, "w") as fd:
                bboxs = preds[idx]
                file_name = os.path.basename(save_name)[:-4] + "\n"
                bboxs_num = str(len(bboxs)) + "\n"
                fd.write(file_name)
                fd.write(bboxs_num)
                for box in bboxs:
                    x = int(box[0] * float(width))
                    y = int(box[1] * float(height))
                    w = int((box[2] - box[0])* float(width))
                    h = int((box[3] - box[1])* float(height))
                    confidence = str(box[4])
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                    fd.write(line)
            
            idx += 1

    return evaluation(save_path+"/images_"+str_uuid+"/")

def evaluation(pred, gt_path="./data/ground_truth/", iou_thresh=0.5):
    pred = get_preds(pred)
    norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []
    for setting_id in range(3):
        # different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num))
        for i in pbar:
            pbar.set_description('Processing {}'.format(settings[setting_id]))
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred[event_name]
            sub_gt_list = gt_list[i][0]
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):
                pred_info = pred_list[str(img_list[j][0][0])]

                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index-1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

                _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

                pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)

    print("==================== Results ====================")
    print("Easy   Val AP: {}".format(aps[0]))
    print("Medium Val AP: {}".format(aps[1]))
    print("Hard   Val AP: {}".format(aps[2]))
    print("=================================================")
    return aps


def bbox_overlaps(boxes_c, query_boxes_c):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    boxes_c = np.expand_dims(boxes_c, axis=1)
    query_boxes_c = np.expand_dims(query_boxes_c, axis=0)

    xx1 = np.maximum(boxes_c[..., 0], query_boxes_c[..., 0])
    yy1 = np.maximum(boxes_c[..., 1], query_boxes_c[..., 1])
    xx2 = np.minimum(boxes_c[..., 2], query_boxes_c[..., 2])
    yy2 = np.minimum(boxes_c[..., 3], query_boxes_c[..., 3])

    w1 = np.maximum(.0, xx2 - xx1 + 1)
    h1 = np.maximum(.0, yy2 - yy1 + 1)
    inter = w1 * h1
    area1 = (boxes_c[..., 2] - boxes_c[..., 0] + 1) * (boxes_c[..., 3] - boxes_c[..., 1] + 1)
    area2 = (query_boxes_c[..., 2] - query_boxes_c[..., 0] + 1) * (query_boxes_c[..., 3] - query_boxes_c[..., 1] + 1)
    overlaps1 = inter / ((area1 + area2) - inter)

    return overlaps1

def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def read_pred_file(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    # b = lines[0].rstrip('\r\n').split(' ')[:-1]
    # c = float(b)
    # a = map(lambda x: [[float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])] for a in x.rstrip('\r\n').split(' ')], lines)
    boxes = []
    for line in lines:
        line = line.rstrip('\r\n').split(' ')
        if line[0] == '':
            continue
        # a = float(line[4])
        boxes.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])])
    boxes = np.array(boxes)
    # boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            # imgname, _boxes = read_pred_file(imgtxt)
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score)/diff


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

