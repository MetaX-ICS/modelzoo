import numpy as np
import paddle

from .nms import filter_bbox, nms_bboxes, sigmoid, softmax, pd_multiclass_nms



def get_yolov4_bboxes_v1(outputs, batch_size, params):

    channel, height, width = params["input_size"].split(",")

    anchors_list = np.array([float(anch) for anch in params['anchors'].split(",")]).reshape(-1,3,2)

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]

    
    npreds = []
    for idx in range(batch_size):
        bboxes, classes, scores = [], [] ,[]
        for i in range(len(outputs)):
            channels, ny, nx = outputs_size_list[i][0], outputs_size_list[i][1], outputs_size_list[i][2]
            output = np.transpose(outputs[i][idx], (1,2,0))
            output = output.reshape(1, ny, nx, 3, -1)
            anchors = anchors_list[i]

            bbox_confs = sigmoid(output[..., 4:5])
            bbox_cls = sigmoid(output[..., 5:])

            bbox_xy = sigmoid(output[..., :2])
            bbox_wh = np.exp(output[..., 2:4]) * anchors

            yv, xv = np.meshgrid(np.arange(ny),np.arange(nx))
            # grid = np.stack((xv, yv), 2).reshape(1, ny, nx, 1, 2).astype('float32')
            grid = np.stack((yv, xv), 2).reshape(1, ny, nx, 1, 2).astype('float32')

            bbox_xy += grid
            bbox_xy /= (nx, ny)
            bbox_wh /= (int(width), int(height))
            bbox_xy -= (bbox_wh / 2.)
            bbox = np.concatenate((bbox_xy, bbox_wh), axis=-1)

            bboxs, classs, score = filter_bbox(bbox, bbox_confs, bbox_cls)

            bboxes.append(bboxs)
            classes.append(classs)
            scores.append(score)

        bboxes = np.concatenate(bboxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)



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


def get_yolov3_bboxes_v1(outputs, batch_size, params):

    channel, height, width = params["input_size"].split(",")

    anchors_list = np.array([float(anch) for anch in params['anchors'].split(",")]).reshape(-1,3,2)

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    # outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2],outputs_size_list[i][3]) for i in range(len(outputs))]

    
    npreds = []
    for idx in range(batch_size):
        bboxes, classes, scores = [], [] ,[]
        for i in range(len(outputs)):
            _, ny, nx, _ = outputs_size_list[i][0], outputs_size_list[i][1], outputs_size_list[i][2], outputs_size_list[i][3]
            output = np.transpose(outputs[i][idx], (1,2,0,3))
            output = output.reshape(1, ny, nx, 3, -1)
            anchors = anchors_list[i]

            bbox_confs = sigmoid(output[..., 4:5])
            bbox_cls = sigmoid(output[..., 5:])
            # bbox_cls = softmax(output[..., 5:])

            # bbox_xy = sigmoid(output[..., :2])
            # bbox_wh = np.exp(output[..., 2:4]) * anchors
            bbox_xy = sigmoid(output[..., :2])*2- 0.5
            bbox_wh = ((sigmoid(output[..., 2:4])*2) **2) * anchors
            # y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

            yv, xv = np.meshgrid(np.arange(ny),np.arange(nx))
            # grid = np.stack((xv, yv), 2).reshape(1, ny, nx, 1, 2).astype('float32')
            grid = np.stack((yv, xv), 2).reshape(1, ny, nx, 1, 2).astype('float32')

            bbox_xy += grid
            bbox_xy /= (nx, ny)
            bbox_wh /= (int(width), int(height))
            bbox_xy -= (bbox_wh / 2.)
            bbox = np.concatenate((bbox_xy, bbox_wh), axis=-1)

            bboxs, classs, score = filter_bbox(bbox, bbox_confs, bbox_cls)

            bboxes.append(bboxs)
            classes.append(classs)
            scores.append(score)

        bboxes = np.concatenate(bboxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

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


def get_yolov3_mmdeploy_postprocess(outputs, batch_size, params):
    channel, height, width = params["input_size"].split(",")

    anchors_list = np.array([float(anch) for anch in params['anchors'].split(",")]).reshape(-1,3,2)

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs.reverse()
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]
    
    
    npreds = []
    for idx in range(batch_size):
        bboxes, classes, scores = [], [] ,[]
        for i in range(len(outputs)):
            _, ny, nx = outputs_size_list[i][0], outputs_size_list[i][1], outputs_size_list[i][2]
            output = np.transpose(outputs[i][idx], (1,2,0))
            output = output.reshape(1, ny, nx, 3, -1)
            anchors = anchors_list[i]

            bbox_confs = sigmoid(output[..., 4:5])
            bbox_cls = sigmoid(output[..., 5:])
            # bbox_cls = softmax(output[..., 5:])

            bbox_xy = sigmoid(output[..., :2])
            bbox_wh = np.exp(output[..., 2:4]) * anchors
            # bbox_xy = sigmoid(output[..., :2])*2- 0.5
            # bbox_wh = ((sigmoid(output[..., 2:4])*2) **2) * anchors

            yv, xv = np.meshgrid(np.arange(ny),np.arange(nx))
            # grid = np.stack((xv, yv), 2).reshape(1, ny, nx, 1, 2).astype('float32')
            grid = np.stack((yv, xv), 2).reshape(1, ny, nx, 1, 2).astype('float32')

            bbox_xy += grid
            bbox_xy /= (nx, ny)
            bbox_wh /= (int(width), int(height))
            bbox_xy -= (bbox_wh / 2.)
            bbox = np.concatenate((bbox_xy, bbox_wh), axis=-1)

            bboxs, classs, score = filter_bbox(bbox, bbox_confs, bbox_cls, 0.01)

            bboxes.append(bboxs)
            classes.append(classs)
            scores.append(score)

        bboxes = np.concatenate(bboxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)



        nbboxes, nclasses, nscores = [], [], []
        for cl in set(classes):
            inds = np.where(classes == cl)
            b = bboxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = nms_bboxes(b, s, 0.6)

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

def get_yolov2_bboxes(outputs, batch_size, params):
    ##"anchors":"1.3221,1.73145,3.19275,4.00944,5.05587,8.09892,9.47112,4.84053,11.2364,10.0071",
    ##"anchors":"0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828",
    channel, height, width = params["input_size"].split(",")
    anchors_list = np.array([float(anch) for anch in params['anchors'].split(",")]).reshape(-1,5,2)
    #anchors_list = np.array(float(anch) for anch in params['anchors'].split(","))
    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2]) for i in range(len(outputs))]

        
    npreds = []
    for idx in range(batch_size):
        bboxes, classes, scores = [], [] ,[]
        for i in range(len(outputs)):
            _, ny, nx = outputs_size_list[i][0], outputs_size_list[i][1], outputs_size_list[i][2]
            output = np.transpose(outputs[i][idx],(1, 2, 0))
            output = output.reshape(1, ny, nx, 5, -1)
            anchors = anchors_list[i]

            bbox_confs = sigmoid(output[..., 4:5])
            bbox_cls = sigmoid(output[..., 5:])
            # bbox_cls = softmax(output[..., 5:])

            bbox_xy = sigmoid(output[..., :2])
            bbox_wh = np.exp(output[..., 2:4]) * anchors
            # bbox_xy = sigmoid(output[..., :2])*2- 0.5
            # bbox_wh = ((sigmoid(output[..., 2:4])*2) **2) * anchors
            # y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

            yv, xv = np.meshgrid(np.arange(ny),np.arange(nx))
            # grid = np.stack((xv, yv), 2).reshape(1, ny, nx, 1, 2).astype('float32')
            grid = np.stack((yv, xv), 2).reshape(1, ny, nx, 1, 2).astype('float32')

            bbox_xy += grid
            bbox_xy /= (nx, ny)
            bbox_wh /= (nx, ny)
            bbox_xy -= (bbox_wh / 2.)
            bbox = np.concatenate((bbox_xy, bbox_wh), axis=-1)

            bboxs, classs, score = filter_bbox(bbox, bbox_confs, bbox_cls)

            bboxes.append(bboxs)
            classes.append(classs)
            scores.append(score)

        bboxes = np.concatenate(bboxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)



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

def get_yolov3_tiny_bboxes_v1(outputs, batch_size, params):
    channel, height, width = params["input_size"].split(",")
    outputs = outputs[:1]
    
    anchors_list = np.array([float(anch) for anch in params['anchors'].split(",")]).reshape(-1,3,2)
    outputs_size = params['outputs_size_T4'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1]) for i in range(len(outputs))]

    
    npreds = []
    for idx in range(batch_size):
        bboxes, classes, scores = [], [] ,[]
        for i in range(len(outputs)):
            # _, ny, nx, _ = outputs_size_list[i][0], outputs_size_list[i][1], outputs_size_list[i][2], outputs_size_list[i][3]
            # output = np.transpose(outputs[i][idx], (1,2,0,3))
            # output = output.reshape(1, ny, nx, 3, -1)
            # anchors = anchors_list[i]
            output = outputs[i][idx]

            bbox_confs = output[..., 4:5]
            bbox_cls = output[..., 5:]

            bbox_xy = output[..., :2]
            bbox_wh = output[..., 2:4]

            bbox_xy /= (int(width), int(height))
            bbox_wh /= (int(width), int(height))
            bbox_xy -= (bbox_wh / 2.)
            bbox = np.concatenate((bbox_xy, bbox_wh), axis=-1)

            bboxs, classs, score = filter_bbox(bbox, bbox_confs, bbox_cls)

            bboxes.append(bboxs)
            classes.append(classs)
            scores.append(score)

        bboxes = np.concatenate(bboxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)



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

def get_yolov4_tiny_bboxes(outputs, batch_size, params):

    channel, height, width = params["input_size"].split(",")
    outputs = outputs[0:2]

    #anchors_list = np.array([float(anch) for anch in params['anchors'].split(",")]).reshape(-1,3,2)
    outputs_size = params['outputs_size_T4'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1]) for i in range(len(outputs))]

    
    npreds = []
    for idx in range(batch_size):
        bboxes, classes, scores = [], [] ,[]
        for i in range(len(outputs)):
          #  _, ny, nx, _ = outputs_size_list[i][0], outputs_size_list[i][1], outputs_size_list[i][2], outputs_size_list[i][3]
          #  output = np.transpose(outputs[i][idx], (1,2,0,3))
          #  output = output.reshape(1, ny, nx, 3, -1)
          #  anchors = anchors_list[i]

            bbox_confs = outputs[1]
            bbox_x1y1 = outputs[0][..., :2]
            bbox_x2x2 = outputs[0][..., 2:4]
            # bbox_cls = outputs[0]
            # bbox_cls = softmax(output[..., 5:])

            # bbox_xy = sigmoid(output[..., :2])
            # bbox_wh = np.exp(output[..., 2:4]) * anchors
           # bbox_xy = sigmoid(output[..., :2])*2- 0.5
           # bbox_wh = ((sigmoid(output[..., 2:4])*2) **2) * anchors
            # y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

           # yv, xv = np.meshgrid(np.arange(ny),np.arange(nx))
            # grid = np.stack((xv, yv), 2).reshape(1, ny, nx, 1, 2).astype('float32')
           # grid = np.stack((yv, xv), 2).reshape(1, ny, nx, 1, 2).astype('float32')

            # bbox_xy += grid
            # bbox_xy /= (int(width), int(height))
            # bbox_wh /= (int(width), int(height))
            # bbox_xy -= (bbox_wh / 2.)
            bbox_wh = bbox_x2x2 - bbox_x1y1
            bbox = np.concatenate((bbox_x1y1, bbox_wh), axis=-1)

            bboxs, classs, score = filter_bbox(bbox, bbox_confs, 1.0)

            bboxes.append(bboxs)
            classes.append(classs)
            scores.append(score)

        bboxes = np.concatenate(bboxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)



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


def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

def get_yolox_bboxes(outputs, batch_size, params):
    channel, height, width = params["input_size"].split(",")
    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1]) for i in range(len(outputs))]
   
    npreds = []
    npreds = []
    for idx in range(batch_size):
        bboxes, classes, scores = [], [] ,[]
        for i in range(len(outputs)):
            output = outputs[0][idx]

            output = demo_postprocess(output, (int(width), int(height)))

            bbox_xy = output[..., :2]
            bbox_wh = output[..., 2:4]
            bbox_confs = output[..., 4:5]
            bbox_cls = output[..., 5:]

            bbox_xy /= (int(width), int(height))
            bbox_wh /= (int(width), int(height))
            bbox_xy -= (bbox_wh / 2.)
            # bbox_wh = bbox_xy - bbox_wh
            bbox = np.concatenate((bbox_xy, bbox_wh), axis=-1)

            bboxs, classs, score = filter_bbox(bbox, bbox_cls, bbox_confs)

            bboxes.append(bboxs)
            classes.append(classs)
            scores.append(score)

        bboxes = np.concatenate(bboxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)



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

def get_yolov5_bboxes(outputs, batch_size, params):
    channel, height, width = params["input_size"].split(",")
    outputs = outputs[:1]
    
    # anchors_list = np.array([float(anch) for anch in params['anchors'].split(",")]).reshape(-1,3,2)
    outputs_size = params['outputs_size_T4'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1]) for i in range(len(outputs))]

    
    npreds = []
    for idx in range(batch_size):
        bboxes, classes, scores = [], [] ,[]
        for i in range(len(outputs)):

            output = outputs[i][idx]

            bbox_confs = output[..., 4:5]
            bbox_cls = output[..., 5:]

            bbox_xy = output[..., :2]
            bbox_wh = output[..., 2:4]

            bbox_xy /= (int(width), int(height))
            bbox_wh /= (int(width), int(height))
            bbox_xy -= (bbox_wh / 2.)
            bbox = np.concatenate((bbox_xy, bbox_wh), axis=-1)

            bboxs, classs, score = filter_bbox(bbox, bbox_confs, bbox_cls, 0.001)

            bboxes.append(bboxs)
            classes.append(classs)
            scores.append(score)

        bboxes = np.concatenate(bboxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)



        nbboxes, nclasses, nscores = [], [], []
        for cl in set(classes):
            inds = np.where(classes == cl)
            b = bboxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = nms_bboxes(b, s, 0.6)

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

def get_yolov5_bboxes_v1(outputs, batch_size, params):
    channel, height, width = params["input_size"].split(",")
    outputs = outputs[1:4]
    
    anchors_list = np.array([float(anch) for anch in params['anchors'].split(",")]).reshape(-1,3,2)
    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2],outputs_size_list[i][3]) for i in range(len(outputs))]

    
    npreds = []
    for idx in range(batch_size):
        bboxes, classes, scores = [], [] ,[]
        for i in range(len(outputs)):
            _, ny, nx, _ = outputs_size_list[i][0], outputs_size_list[i][1], outputs_size_list[i][2], outputs_size_list[i][3]
            output = np.transpose(sigmoid(outputs[i][idx]), (1,2,0,3))
            output = output.reshape(1, ny, nx, 3, -1)
            anchors = anchors_list[i]

            bbox_confs = output[..., 4:5]
            bbox_cls = output[..., 5:]

            bbox_xy = output[..., :2]*2- 0.5
            bbox_wh = ((output[..., 2:4]*2) **2) * anchors

            yv, xv = np.meshgrid(np.arange(ny),np.arange(nx))
            # grid = np.stack((xv, yv), 2).reshape(1, ny, nx, 1, 2).astype('float32')
            grid = np.stack((yv, xv), 2).reshape(1, ny, nx, 1, 2).astype('float32')

            bbox_xy += grid
            bbox_xy /= (nx, ny)
            bbox_wh /= (int(width), int(height))
            bbox_xy -= (bbox_wh / 2.)
            bbox = np.concatenate((bbox_xy, bbox_wh), axis=-1)

            bboxs, classs, score = filter_bbox(bbox, bbox_confs, bbox_cls, 0.001)

            bboxes.append(bboxs)
            classes.append(classs)
            scores.append(score)

        bboxes = np.concatenate(bboxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)



        nbboxes, nclasses, nscores = [], [], []
        for cl in set(classes):
            inds = np.where(classes == cl)
            b = bboxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = nms_bboxes(b, s, 0.6)

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

def get_yolov7_bboxes_v1(outputs, batch_size, params):
    channel, height, width = params["input_size"].split(",")
    # outputs = outputs[1:4]
    
    anchors_list = np.array([float(anch) for anch in params['anchors'].split(",")]).reshape(-1,3,2)
    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2],outputs_size_list[i][3]) for i in range(len(outputs))]

    
    npreds = []
    for idx in range(batch_size):
        bboxes, classes, scores = [], [] ,[]
        for i in range(len(outputs)):
            _, ny, nx, _ = outputs_size_list[i][0], outputs_size_list[i][1], outputs_size_list[i][2], outputs_size_list[i][3]
            output = np.transpose(sigmoid(outputs[i][idx]), (1,2,0,3))
            output = output.reshape(1, ny, nx, 3, -1)
            anchors = anchors_list[i]

            bbox_confs = output[..., 4:5]
            bbox_cls = output[..., 5:]

            bbox_xy = output[..., :2]*2- 0.5
            bbox_wh = ((output[..., 2:4]*2) **2) * anchors

            yv, xv = np.meshgrid(np.arange(ny),np.arange(nx))
            # grid = np.stack((xv, yv), 2).reshape(1, ny, nx, 1, 2).astype('float32')
            grid = np.stack((yv, xv), 2).reshape(1, ny, nx, 1, 2).astype('float32')

            bbox_xy += grid
            bbox_xy /= (nx, ny)
            bbox_wh /= (int(width), int(height))
            bbox_xy -= (bbox_wh / 2.)
            bbox = np.concatenate((bbox_xy, bbox_wh), axis=-1)

            bboxs, classs, score = filter_bbox(bbox, bbox_confs, bbox_cls, 0.001)

            bboxes.append(bboxs)
            classes.append(classs)
            scores.append(score)

        bboxes = np.concatenate(bboxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)



        nbboxes, nclasses, nscores = [], [], []
        for cl in set(classes):
            inds = np.where(classes == cl)
            b = bboxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = nms_bboxes(b, s, 0.65)

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