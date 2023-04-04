#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import os
import sys
sys.setrecursionlimit(100000)
import numpy as np
from tqdm import trange

import onnxruntime as ort

from common import get_params, preprocess_test, get_FPS_multithread, get_FPS_multithread_v1, \
    check_accuracy_multithread, get_EP, model_convert, check_model

import logging
logging.basicConfig(level=logging.INFO #设置日志输出格式
                    ,filename="./DET_LOGGING_ORT.log" 
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

    batch_size = int(batchsize)
    precision = precision
    th_num = int(th_num)
    FPS = 0

    image_list = preprocess_test(params['image_T4'])#[:100]
    print("load {} images".format(len(image_list)))

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
        if "maca" not in EP:
            FPS = get_FPS_multithread(sess, batch_size, precision, image_list, params, th_num)
        else:
            FPS = get_FPS_multithread_v1(sess, batch_size, precision, image_list, params, th_num)

        if task == "fpsonly":
            print("detec_{}_bs{}_prec{} FPS : {:.3f}".format(modelname.replace("/", "_") ,batch_size, precision, FPS))
            logging.info("detec_{}_bs{}_prec{} FPS : {:.3f}".format(modelname.replace("/", "_") ,batch_size, precision, FPS))
            return
    check_accuracy_multithread(sess, batch_size, precision, params, modelname, test_set=image_list)
    # check_accuracy(sess, batch_size, precision, params, modelname, test_set=image_list)
    
    print("detec_{}_bs{}_prec{} FPS : {:.3f}".format(modelname.replace("/", "_") ,batch_size, precision, FPS))
    logging.info("detec_{}_bs{}_prec{} FPS : {:.3f}".format(modelname.replace("/", "_") ,batch_size, precision, FPS))





if __name__ == '__main__':
    modelname = sys.argv[1]
    batchsize = sys.argv[2]
    precision = sys.argv[3]
    task = sys.argv[4]
    model_path = sys.argv[5] if len(sys.argv) > 5 else "./"
    EP = sys.argv[6] if len(sys.argv) > 6 else "maca"
    th_num = sys.argv[7] if len(sys.argv) > 7 else "16"
    main(modelname,batchsize,precision,task,model_path, EP, th_num) 


                     
