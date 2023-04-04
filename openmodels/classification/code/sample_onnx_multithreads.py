#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import time
import os
import sys

import onnxruntime as ort

from common import get_params, preprocess_test, get_FPS_multithread, get_FPS_multithread_v1, \
    check_accuracy_multithread, get_EP, model_convert, check_model

import logging
logging.basicConfig(level=logging.INFO #设置日志输出格式
                    ,filename="./CLS_LOGGING_ORT.log" 
                    ,format="%(asctime)s - %(name)s - %(levelname)-9s: %(message)s" 
                    ,datefmt="%Y-%m-%d %H:%M:%S" #时间输出的格式
                    )


def main(modelname,batchsize,precision, task="normal", modelfile="./",EP="maca", th_num="8"):
    params = get_params(modelname)
    model_path = check_model(modelfile, modelname, params)
    if not os.path.exists(model_path):
        print("cannot find {}".format(model_path))
        return
    print("running model: {}, precision is: {}".format(model_path, precision))

    cls_num = int(params['cls_num'])
    batch_size = int(batchsize)
    precision = precision
    th_num = int(th_num)
    FPS = 0

    image_list = preprocess_test(params['label_T4'])#[:100]
    print("load {} images".format(len(image_list)))

    # Set graph optimization level
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

    EP = EP.split("+")
    provider_list = get_EP(EP)
    start = time.time()
    model_path = model_convert(modelname, model_path, precision, params, task)
    if model_path != "":
        sess = ort.InferenceSession(model_path,providers=provider_list)  
        end = time.time()
        print("load {} model -- {}, cost {:.2f}s".format(precision, model_path, end - start))
    else:
        return 

    if task != "fpsclosed":
    # FPS = get_FPS(sess, batch_size, precision, test_set=inputs,)## 64kernel
        if "maca" not in EP:
            FPS = get_FPS_multithread(sess, batch_size, precision, image_list, params, th_num)
        else:
            FPS = get_FPS_multithread_v1(sess, batch_size, precision, image_list, params, th_num)## 64kernel
        
        if task == "fpsonly":
            print("class_{}_bs{}_prec{}_ort FPS : {:.2f}".format(modelname.replace("/", "_"), batch_size, precision, FPS))
            logging.info("class_{}_bs{}_prec{}_ort FPS : {:.2f}".format(modelname.replace("/", "_"), batch_size, precision, FPS))
            return
    top1, top5 = check_accuracy_multithread(sess, 1, precision, cls_num, image_list, params) 
    # top1, top5 = check_accuracy_v1(sess, batch_size, precision, cls_num, image_list, params) 

    print("class_{}_bs{}_prec{} result: top1 : {:.2f}%, top5 : {:.2f}%".format(modelname.replace("/", "_"), batch_size, precision, top1, top5))
    # print("class_{}_bs{}_prec{} result: top5 : {:.2f}%".format(modelname.replace("/", "_"), batch_size, precision, top5*100))
    logging.info("class_{}_bs{}_prec{} result: top1 : {:.2f}%, top5 : {:.2f}%".format(modelname.replace("/", "_"), batch_size, precision, top1, top5))
    # logging.info("class_{}_bs{}_prec{} result: top5 : {:.2f}%".format(modelname.replace("/", "_"), batch_size, precision, top5*100))

    print("class_{}_bs{}_prec{}_ort FPS : {:.3f}".format(modelname.replace("/", "_"), batch_size, precision, FPS))
    logging.info("class_{}_bs{}_prec{}_ort FPS : {:.3f}".format(modelname.replace("/", "_"), batch_size, precision, FPS))


if __name__ == '__main__':
    modelname = sys.argv[1]
    batchsize = sys.argv[2]
    precision = sys.argv[3]
    task = sys.argv[4] if len(sys.argv) > 4 else "normal"
    model_path = sys.argv[5] if len(sys.argv) > 5 else "./"
    EP = sys.argv[6] if len(sys.argv) > 6 else "maca"
    th_num = sys.argv[7] if len(sys.argv) > 7 else "16"
    main(modelname,batchsize,precision,task,model_path, EP, th_num) 

    # logging.info("")
    # logging.info("")
    # logging.info("")
    # logging.info("")
    # logging.info("")

    
                     
