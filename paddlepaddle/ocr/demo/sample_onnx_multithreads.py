#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import os
import sys
import numpy as np
from tqdm import trange

import onnxruntime as ort

from common import get_params, preprocess_test, get_FPS_multithread, get_FPS_multithread_v1, \
    check_accuracy_multithread, get_EP, model_convert, check_model, get_cmd_output_with_list
import logging
logging.basicConfig(level=logging.INFO #设置日志输出格式
                    ,filename="./OCR_LOGGING_ORT.log" 
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

    image_list = preprocess_test(params['image_T4'])#[:10]
    print("load {} images".format(len(image_list))) 
    
    memory_usage_start = get_cmd_output_with_list("mx-smi --show-memory|awk 'NR==13'")

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
        if "maca" not in EP:
            FPS = get_FPS_multithread(sess, batch_size, precision, image_list, params, th_num)
        else:
            FPS, dla_utils = get_FPS_multithread_v1(sess, batch_size, precision, image_list, params, th_num)
            
        st_idx = len(dla_utils) // 3 * 2
        avg_utils = [int(util.replace("%", "")) for util in dla_utils[st_idx:]]
        print("DLA avg utils is {:.2f}%".format(sum(avg_utils)/len(avg_utils)))
        # print(dla_utils)
        
        memory_usage_end = get_cmd_output_with_list("mx-smi --show-memory|awk 'NR==13'")
        memory_usage = int(memory_usage_end.replace("KB", "")) - int(memory_usage_start.replace("KB", ""))


        if task == "fpsonly":
            print("OCR_{}_bs{}_prec{} ORTFPS: {:.3f}".format(modelname.replace("/", "_") ,batch_size, precision, FPS))
            logging.info("OCR_{}_bs{}_prec{} ORTFPS: {:.3f}".format(modelname.replace("/", "_") ,batch_size, precision, FPS))
            print("memory usage is {:.2f} MB".format(memory_usage/1024))
            return
    # check_accuracy(sess, batch_size, precision, params, modelname, test_set=image_list)
    check_accuracy_multithread(sess, batch_size, precision, params, modelname, test_set=image_list)
    memory_usage_end = get_cmd_output_with_list("mx-smi --show-memory|awk 'NR==13'")
    memory_usage = int(memory_usage_end.replace("KB", "")) - int(memory_usage_start.replace("KB", ""))
    print("memory usage is {:.2f}MB".format(memory_usage/1024))

    print("OCR_{}_bs{}_prec{} ORTFPS: {:.3f}".format(modelname.replace("/", "_") ,batch_size, precision, FPS))
    logging.info("OCR_{}_bs{}_prec{} ORTFPS: {:.3f}".format(modelname.replace("/", "_") ,batch_size, precision, FPS))


    print("Run with precision {}, batchsize {} ".format(precision, batch_size))
    # logging.info("Run with precision {}, batchsize {} ".format(precision, batch_size))


if __name__ == '__main__':
    modelname = sys.argv[1]
    batchsize = sys.argv[2]
    precision = sys.argv[3]

    task = sys.argv[4]      
    model_path = sys.argv[5] if len(sys.argv) > 5 else "./"
    EP = sys.argv[6] if len(sys.argv) > 6 else "maca"
    th_num = sys.argv[7] if len(sys.argv) > 7 else "16"
    main(modelname,batchsize,precision,task,model_path, EP, th_num) 


                     
