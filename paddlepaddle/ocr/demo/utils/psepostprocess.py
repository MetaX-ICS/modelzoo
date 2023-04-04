import numpy as np
import cv2
import queue as Queue
from .db_postprocess import DBPostProcess

def sigmoid(x):
    return 1/(1+np.exp(-x))

def pypse(kernals, min_area):
    kernal_num = len(kernals)
    pred = np.zeros(kernals[0].shape, dtype='int32')

    label_num, label = cv2.connectedComponents(kernals[kernal_num - 1], connectivity=4)
    # show raw_label
    # label[np.nonzero(label)]=255
    # cv2.imwrite('temp.jpg',label)

    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0

    queue = Queue.Queue(maxsize=0)
    next_queue = Queue.Queue(maxsize=0)
    points = np.array(np.where(label > 0)).transpose((1, 0))

    for point_idx in range(points.shape[0]):
        x, y = points[point_idx, 0], points[point_idx, 1]
        l = label[x, y]
        queue.put((x, y, l))
        pred[x, y] = l

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    for kernal_idx in range(kernal_num - 2, -1, -1):
        kernal = kernals[kernal_idx].copy()
        while not queue.empty():
            (x, y, l) = queue.get()

            is_edge = True
            for j in range(4):
                tmpx = x + dx[j]
                tmpy = y + dy[j]
                if tmpx < 0 or tmpx >= kernal.shape[0] or tmpy < 0 or tmpy >= kernal.shape[1]:
                    continue
                if kernal[tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
                    continue

                queue.put((tmpx, tmpy, l))
                pred[tmpx, tmpy] = l
                is_edge = False
            if is_edge:
                next_queue.put((x, y, l))

        # kernal[pred > 0] = 0
        queue, next_queue = next_queue, queue

        # points = np.array(np.where(pred > 0)).transpose((1, 0))
        # for point_idx in range(points.shape[0]):
        #     x, y = points[point_idx, 0], points[point_idx, 1]
        #     l = pred[x, y]
        #     queue.put((x, y, l))

    return pred


def get_pse_post(outputs, batch_size, params, content):

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    outputs = [outputs[i].reshape(-1,outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]
    
    preds = []

    for idx in range(batch_size):
        output = outputs[0][idx]

        # output = output[0]
        score = sigmoid(output[0, :, :])
        output = (np.sign(output - 1.0) + 1) / 2
        text = output[0, :, :]
        kernels = output[0:7, :, :] * text

        score = score.astype(np.float32)
        text = text.astype(np.uint8)
        kernels = kernels.astype(np.uint8)

        pred = pypse(kernels, 5)
        # print(pred.shape)

        # scale = (org_img.shape[0] * 1.0 / pred.shape[0], org_img.shape[1] * 1.0 / pred.shape[1])
        scale = (1280 * 1.0 / pred.shape[1], 720 * 1.0 / pred.shape[0])
        label = pred
        label_num = np.max(label) + 1
        bboxes = []
        for i in range(1, label_num):
            points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < 800:
                continue

            score_i = np.mean(score[label == i])
            if score_i < 0.93:
                continue

            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect) * scale
            # bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
        preds.append(bboxes)
    return preds

def get_dbnet_post(outputs, batch_size, params, content):

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    outputs = [outputs[i].reshape(-1,outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]
    
    preds = []
    db_processs = DBPostProcess(thresh=0.3,
                                box_thresh=0.6,
                                max_candidates=1000,
                                unclip_ratio=1.5)

    for idx in range(batch_size):
        output = outputs[0][idx].copy()

        pred_dict = {}
        pred_dict["maps"] = np.expand_dims(output, axis=0)
        bboxs = db_processs(pred_dict, np.array([[720.,1280.,736/720.,1.]]))
        bboxes = [box.reshape(-1) for box in bboxs[0]]

        preds.append(bboxes)
    return preds

