import time
import os
import sys

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from .tools import preprocess_test


class iouEval:

    def __init__(self, nClasses, ignoreIndex=19):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses>ignoreIndex else -1 #if ignoreIndex is larger than nClasses, consider no ignoreIndex
        self.reset()

    def reset (self):
        classes = self.nClasses if self.ignoreIndex==-1 else self.nClasses-1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()        

    def addBatch(self, x, y):   #x=preds, y=targets
        #sizes should be "batch_size x nClasses x H x W"
        
        #print ("X is cuda: ", x.is_cuda)
        #print ("Y is cuda: ", y.is_cuda)

        if (x.is_cuda or y.is_cuda):
            x = x.cuda()
            y = y.cuda()

        #if size is "batch_size x 1 x H x W" scatter to onehot
        if (x.size(1) == 1):
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))  
            if x.is_cuda:
                x_onehot = x_onehot.cuda()
            x_onehot.scatter_(1, x, 1).float()
        else:
            x_onehot = x.float()

        if (y.size(1) == 1):
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3))
            if y.is_cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

        if (self.ignoreIndex != -1): 
            ignores = y_onehot[:,self.ignoreIndex].unsqueeze(1)
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores=0

        #print(type(x_onehot))
        #print(type(y_onehot))
        #print(x_onehot.size())
        #print(y_onehot.size())

        tpmult = x_onehot * y_onehot    #times prediction and gt coincide is 1
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fpmult = x_onehot * (1-y_onehot-ignores) #times prediction says its that class and gt says its not (subtracting cases when its ignore label!)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fnmult = (1-x_onehot) * (y_onehot) #times prediction says its not that class and gt says it is
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze() 

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou), iou     #returns "iou mean", "iou per class"


class Relabel:
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel
    
    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor), "tensor need to be longtensor"
        tensor[tensor == self.olabel] = self.nlabel
        return tensor

class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


def comute_citys_scores_v1(predns, params, modelname, datanums):
    channel, height, width = params["input_size"].split(",")
    ann_dir="./data/Cityscapes_data/gtFine/val/"
    iouEvalVal = iouEval(20)
    label_list = preprocess_test(params["image_T4"])
    trans_preprocess = transforms.Compose([
            transforms.Resize((int(height), int(width)), Image.NEAREST),
            ToLabel(),
            Relabel(255, 19),   #ignore label to 19
        ])
    for i in range(datanums):
        
        label_name = label_list[i][0].split("/")
        label_file = ann_dir + label_name[-2]+"/" + label_name[-1].replace("leftImg8bit","gtFine_labelTrainIds")
        label_tensor = trans_preprocess(Image.open(label_file).convert("P")).unsqueeze(0)#.cuda()

        preds = torch.from_numpy(predns[i]).unsqueeze(0)#.cuda()
        
        iouEvalVal.addBatch(preds.unsqueeze(1).data, label_tensor)
    

    iouVal, iou_classes = iouEvalVal.getIoU()
    # iou_classes_str = []
    # for i in range(iou_classes.size(0)):
    #     iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
    #     iou_classes_str.append(iouStr)

    # print("---------------------------------------")
    # print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes[0], "Road")
    print(iou_classes[1], "sidewalk")
    print(iou_classes[2], "building")
    print(iou_classes[3], "wall")
    print(iou_classes[4], "fence")
    print(iou_classes[5], "pole")
    print(iou_classes[6], "traffic light")
    print(iou_classes[7], "traffic sign")
    print(iou_classes[8], "vegetation")
    print(iou_classes[9], "terrain")
    print(iou_classes[10], "sky")
    print(iou_classes[11], "person")
    print(iou_classes[12], "rider")
    print(iou_classes[13], "car")
    print(iou_classes[14], "truck")
    print(iou_classes[15], "bus")
    print(iou_classes[16], "train")
    print(iou_classes[17], "motorcycle")
    print(iou_classes[18], "bicycle")
    print("=======================================")
    # iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouVal, "%")

    return iouVal