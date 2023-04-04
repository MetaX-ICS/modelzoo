import numpy as np
try:
    import paddle._legacy_C_ops as C_ops
except:
    import paddle._C_ops as C_ops

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

def nms_bboxes(boxes, scores, thr = 0.45): 
    boxes[:, 0] = boxes[:,0].clip(0., 1.)
    boxes[:, 1] = boxes[:,1].clip(0., 1.)
    boxes[:, 2] = boxes[:,2].clip(0., 1.)
    boxes[:, 3] = boxes[:,3].clip(0., 1.)

    x = boxes[:,0]
    y = boxes[:,1]
    w = boxes[:,2]
    h = boxes[:,3]

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

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= thr)[0]
        order = order[inds+1]
    
    keep = np.array(keep)
    return keep[:100]

def filter_bbox(bboxes, bbox_confs, bbox_cls, thr = 0.01, topk=1000):

    bbox_confs_score = bbox_confs * bbox_cls
    classes = np.argmax(bbox_confs_score, axis=-1)
    # bbox_class = np.expand_dims(bbox_class, axis=-1)
    scores = np.max(bbox_confs_score, axis=-1)

    pos = np.where(scores >= thr)
    bboxes = bboxes[pos]  
    classes = classes[pos]
    scores = scores[pos]
    
    if topk < scores.size:
        index = np.argsort(-scores)[:topk]
        bboxes = bboxes[index]  
        classes = classes[index]
        scores = scores[index]
 
    return bboxes, classes, scores

def pd_multiclass_nms(bboxes, scores, score_thr, nms_thr, keep_top_k, cls_num=80, nms_top_k=1000):
    attrs = ("background_label",cls_num,"score_threshold",score_thr,
            "nms_top_k",nms_top_k, "nms_threshold", nms_thr,
            "keep_top_k",keep_top_k, "nms_eta", 1.0, "normalized", True)

    output, _, _ = C_ops.multiclass_nms3(bboxes, scores, None, *attrs)

    return output

def pd_matrix_nms(bboxes, scores, score_thr, nms_thr, keep_top_k, cls_num=80, nms_top_k=1000):
    attrs = ("background_label",cls_num,"score_threshold",score_thr,
            "nms_top_k",nms_top_k, "post_threshold", nms_thr,"gaussian_sigma", 2.,
            "keep_top_k",keep_top_k, "use_gaussian", False, "normalized", False)

    output, _, _ = C_ops.matrix_nms(bboxes, scores, *attrs)

    return output