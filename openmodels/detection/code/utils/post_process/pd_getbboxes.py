import numpy as np
import paddle

from .nms import filter_bbox, nms_bboxes, sigmoid, softmax, pd_multiclass_nms, pd_matrix_nms

def distance2bbox(points, dist):
    # lt, rb = paddle.split(dist,2,-1)
    x1y1 = -dist[...,:2] + points
    x2y2 = dist[...,2:]  + points

    # bbox_xy = - dist[:,:2] + points
    # bbox_wh = dist[:,:2]  + dist[:,2:] 

    out_bbox = np.concatenate([x1y1, x2y2], -1)

    return out_bbox


def ppyolo_pd_process(outputs, batch_size, params, contents):
    # anchor_points1, stride_tensor1 = _generate_anchors(size)
    channel, height, width = params["input_size"].split(",")
    
    for idx in range(batch_size):
        pred_scores, pred_dist, anchor_points, stride_tensor = outputs
        pred_bboxes = distance2bbox(anchor_points[None], pred_dist.transpose([0, 2, 1]))
        pred_bboxes *= stride_tensor
        pred_bboxes /= (int(width), int(height), int(width), int(height))
        
        preds = pd_multiclass_nms(paddle.to_tensor(pred_bboxes), paddle.to_tensor(pred_scores), 0.01, 0.7, 300)

    return preds.numpy()

def get_ppyolo_pd_bboxes(outputs, batch_size, params, contents):
    # outputs.reverse()
    # channel, height, width = params["input_size"].split(",")

    anchors_list = np.array([float(anch) for anch in params['anchors'].split(",")]).reshape(-1,6)

    npreds = []
    for idx in range(batch_size):
        yolo_boxes, yolo_scores = [], [] 
        content = contents[idx]
        for i in range(len(outputs)):
            head_out = paddle.to_tensor(outputs[i])
            # ori_shape = paddle.to_tensor([[int(height), int(weight)]], dtype="int32")
            ori_shape = paddle.to_tensor([[int(content[3]), int(content[2])]], dtype="int32")
            boxes, scores = paddle.vision.ops.yolo_box(
                head_out,
                ori_shape,
                list(anchors_list[i]),
                80,
                0.01,
                32 // 2**i,
                True,
                scale_x_y=float(params["scale_x_y"])
            )
            # boxes /= paddle.to_tensor([float(width), float(height),float(width), float(height)])
            yolo_boxes.append(boxes)
            yolo_scores.append(paddle.transpose(scores,perm=[0,2,1]))

        yolo_boxes = paddle.concat(yolo_boxes, axis=1)
        yolo_scores = paddle.concat(yolo_scores, axis=2)

        preds = pd_matrix_nms(yolo_boxes, yolo_scores, 0.01, 0.01, 100, -1, -1)
        # preds = pd_multiclass_nms(yolo_boxes, yolo_scores, 0.005, 0.45, 100)
        if preds.size != 0:
            preds[...,2:6] /= paddle.to_tensor([float(content[2]), float(content[3]),float(content[2]), float(content[3])])
        npreds.append(preds.numpy())

    return npreds


def get_yolov3_pd_bboxes(outputs, batch_size, params, contents):
    # outputs.reverse()
    # channel, height, width = params["input_size"].split(",")

    anchors_list = np.array([float(anch) for anch in params['anchors'].split(",")]).reshape(-1,6)

    # outputs_size = params['outputs_size'].split("#")
    # outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]

    npreds = []
    for idx in range(batch_size):
        yolo_boxes, yolo_scores = [], [] 
        content = contents[idx]
        for i in range(len(outputs)):
            head_out = paddle.to_tensor(outputs[i])
            # ori_shape = paddle.to_tensor([[int(width), int(height)]], dtype="int32")
            ori_shape = paddle.to_tensor([[int(content[3]), int(content[2])]], dtype="int32")
            boxes, scores = paddle.vision.ops.yolo_box(
                head_out,
                ori_shape,
                list(anchors_list[i]),
                80,
                0.005,
                32 // 2**i,
                True,
                scale_x_y=float(params["scale_x_y"])
            )
            # boxes /= paddle.to_tensor([float(width), float(height),float(width), float(height)])
            yolo_boxes.append(boxes)
            yolo_scores.append(paddle.transpose(scores,perm=[0,2,1]))

        yolo_boxes = paddle.concat(yolo_boxes, axis=1)
        yolo_scores = paddle.concat(yolo_scores, axis=2)

        preds = pd_multiclass_nms(yolo_boxes, yolo_scores, 0.005, 0.45, 100)
        if preds.size != 0:
            preds[...,2:6] /= paddle.to_tensor([float(content[3]), float(content[2]),float(content[3]), float(content[2])])
        npreds.append(preds.numpy())

    return npreds

def get_ppyoloe_pd_bboxes(outputs, batch_size, params, contents):
    channel, height, width = params["input_size"].split(",")

    # outputs_size = params['outputs_size'].split("#")
    # outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]

    npreds = []
    for idx in range(batch_size):
        pred_scores, pred_dist, anchor_points, stride_tensor = outputs
        pred_bboxes = distance2bbox(anchor_points[None], pred_dist)
        pred_bboxes *= stride_tensor
        pred_bboxes /= (int(width), int(height), int(width), int(height))
        
        preds = pd_multiclass_nms(paddle.to_tensor(pred_bboxes), paddle.to_tensor(pred_scores), 0.01, 0.7, 300)
        npreds.append(preds.numpy())
        
    return npreds