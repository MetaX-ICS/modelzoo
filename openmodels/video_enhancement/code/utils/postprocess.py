from .tools import preprocess_test
import numpy as np
from PIL import Image
import math

import logging

def PSNR(pred, gt, shave_border=4):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def compute_PSNR(preds, params, iters, batch_size, precision, modelname):
    inputs = preprocess_test(params['image_T4'])
    for i in range(iters):
        pred = preds[i]
        if len(pred) != 3:
            pred = pred.squeeze()

            bicu_name = inputs[i][0]
            bicu = np.load("./data/" + bicu_name)
            gt_name = bicu_name[:-12] + ".npy"
            gt = np.load("./data/" + gt_name)
            if modelname == "models/ox_dncnn_512x512":
                pred = bicu - pred

            # pred = bicu-pred  ## for dncnn
            psnr_predicted = PSNR(pred, gt)
            psnr_bicubic = PSNR(bicu, gt)

            print("imagename is :", inputs[i])
            print("videoenhance_{}_bs1_prec{} Acc: {}'s PSNR_bicubic_{} : {:.2f}, PSNR_predicted_{} : {:.2f}"\
                .format(modelname.replace("/","_"), precision, inputs[i][0], i,psnr_bicubic, i, psnr_predicted))
            logging.info("videoenhance_{}_bs1_prec{} TAcc: {}'s PSNR_bicubic_{} : {:.2f}, PSNR_predicted_{} : {:.2f}"\
                .format(modelname.replace("/","_"), precision, inputs[i][0], i, psnr_bicubic, i, psnr_predicted))
        else:
            pred = pred.transpose([1, 2, 0])
            bicu_name = inputs[i][0]
            gt_name = bicu_name[:11] + "_GT.bmp"
            # print(gt_name)
            gt = np.array(Image.open("./data/" + gt_name))[:,:,[2,1,0]]
            # print(pred.shape)
            # print(gt.shape)

            psnr_predicted = PSNR(pred, gt)
            print("imagename is :", inputs[i])
            print("videoenhance_{}_bs1_prec{} Acc: {}'s  PSNR_predicted_{} : {:.2f}"\
                .format(modelname.replace("/","_"), precision, inputs[i][0], i, psnr_predicted))
            logging.info("videoenhance_{}_bs1_prec{} Acc: {}'s  PSNR_predicted_{} : {:.2f}"\
                .format(modelname.replace("/","_"), precision, inputs[i][0], i, psnr_predicted))

def get_srimg(type, outputs, batch_size, params):
    return eval(type)(outputs, batch_size, params)

def get_base_post(outputs, batch_size, params):

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2])for i in range(len(outputs))]  #

    npreds = []
    for idx in range(batch_size):
        out_image = outputs[0][idx] #* 255.
        out_image = out_image.round().clip(max=255., min=0)

        npreds.append(out_image)
    return npreds

def get_srres_post(outputs, batch_size, params):
    
    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2])for i in range(len(outputs))]  #

    out_images = []
    for idx in range(batch_size):
        out_image = outputs[0][idx] #* 255.
        out_image = out_image.round().clip(max=255., min=0)

        out_images.append(out_image.transpose([2,0,1]))

    return out_images

def get_srcnn_post(outputs, batch_size, params):

    outputs_size = params['outputs_size'].split("#")
    outputs_size_list = [ [int(size) for size in output_size.split(",")] for output_size in outputs_size]
    # outputs = [outputs[i][:batch_size*outputs_size_list[i][0]*outputs_size_list[i][1]*outputs_size_list[i][2]] for i in range(len(outputs))]
    outputs = [outputs[i].reshape(-1, outputs_size_list[i][0],outputs_size_list[i][1],outputs_size_list[i][2])for i in range(len(outputs))]  #

    npreds = []
    for idx in range(batch_size):
        out_image = outputs[0][idx] * 255.
        out_image = out_image.round().clip(max=255., min=0)

        npreds.append(out_image)
    return npreds