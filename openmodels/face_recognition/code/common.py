#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import time
import json
import os
import onnxruntime as ort
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import onnx
from tqdm import trange
import yaml

from utils import  get_params, preprocess_test, get_images, get_faces, \
    comute_retinaface_map, comute_mobface_acc, comute_arcface_acc, parseList

import logging


def comute_acc(preds, params, save_path, test_set, folds_flags=None, batch_size=1, precision="fp16"):
    dataset_type = "widerface" if "dataset" not in params.keys() else params['dataset']
    if dataset_type == "widerface":
        aps = comute_retinaface_map(preds, params, save_path, len(test_set))
        print("facerecongition_{}_bs{}_prec{} result EasyAP50 : {:.2f}%".format(save_path.replace("/", "_") ,batch_size, precision, aps[0]*100))
        print("facerecongition_{}_bs{}_prec{} result MediumAP50 : {:.2f}%".format(save_path.replace("/", "_") ,batch_size, precision, aps[1]*100))
        print("facerecongition_{}_bs{}_prec{} result HardAP50 : {:.2f}%".format(save_path.replace("/", "_") ,batch_size, precision, aps[2]*100))
        logging.info("facerecongition_{}_bs{}_prec{} result EasyAP50 : {:.2f}%".format(save_path.replace("/", "_") ,batch_size, precision, aps[0]*100))
        logging.info("facerecongition_{}_bs{}_prec{} result MediumAP50 : {:.2f}%".format(save_path.replace("/", "_") ,batch_size, precision, aps[1]*100))
        logging.info("facerecongition_{}_bs{}_prec{} result HardAP50 : {:.2f}%".format(save_path.replace("/", "_") ,batch_size, precision, aps[2]*100))
    elif dataset_type == "lfw_mob":
        acc = comute_mobface_acc(preds, folds_flags)
        print("facerecongition_{}_bs{}_prec{} result lfw accuracy : {:.2f}%".format(save_path.replace("/", "_") ,batch_size, precision, acc*100))
        logging.info("facerecongition_{}_bs{}_prec{} result lfw accuracy : {:.2f}%".format(save_path.replace("/", "_") ,batch_size, precision, acc*100))
    elif dataset_type == "lfw_arc":
        acc, th = comute_arcface_acc(preds, params, test_set)
        print("facerecongition_{}_bs{}_prec{} result lfw accuracy : {:.2f}%, threshold : {}".format(save_path.replace("/", "_") ,batch_size, precision, acc*100, th))
        logging.info("facerecongition_{}_bs{}_prec{} result lfw accuracy : {:.2f}%, threshold : {}".format(save_path.replace("/", "_") ,batch_size, precision, acc*100, th))
    else:
        pass
    
def modify_yaml(yaml_file, params, model_file, export_type="native"):
    # with open(yaml_file, "r") as f:
    #     content = yaml.load(f.read())

    model_path = os.path.split(yaml_file)[0]

    content = {}

    content["dataset"] = {}
    content["dataset"]["batch_size"] = 8
    content["dataset"]["calib_dir"] = params["image_T4"]
    content["dataset"]["calib_num"] = 200
    content["dataset"]["preprocessing"] = {}
    content["dataset"]["preprocessing"]["enable"] = False if params["post_process"]!="get_retinaface_post" else True
    content["dataset"]["preprocessing"]["type"] = "preprocess_v1"
    content["dataset"]["preprocessing"]["attributes"] = {}
    mean = [float(value) for value in params['mean'].split(",")]
    std = [float(value) for value in params['std'].split(",")] if "std" in params.keys() \
        else [1./float(value) for value in params['scale'].split(",")]
    content["dataset"]["preprocessing"]["attributes"]["mean"] = [value if value>1 else value*255. for value in mean] if len(mean) != 1 \
        else [value if value>1 else value*255. for value in mean]*3
    content["dataset"]["preprocessing"]["attributes"]["std"] = [value if value>1 else value*255. for value in std] if len(std) != 1 \
        else [value if value>=1 else value*255. for value in std]*3
    content["dataset"]["preprocessing"]["attributes"]["isreverses"] = True if "reverse_channel" in params.keys() else False
    content["dataset"]["preprocessing"]["attributes"]["resize"] = {}
    size = [int(value) for value in params['input_size'].split(",")]
    content["dataset"]["preprocessing"]["attributes"]["resize"]["keep_ratio"] = False
    content["dataset"]["preprocessing"]["attributes"]["resize"]["to"] = size

    content["import_model"] = model_file
    content["export_model"] = model_path + "/" +  os.path.basename(model_file).replace(".onnx", "_int8_maca.onnx")
    content["export_type"] = "maca"
    if params["post_process"]=="get_arcface_post":
        content["force_advance_quant"] = True

    with open(yaml_file, "w") as fw:
        yaml.dump(content,fw)
    # common.write_yml(content, yaml_file)

    print("yaml file is :"+yaml_file)
    print("+++++++++++++++++++++ modify_yaml is Success ++++++++++++++")


task_list = [i for i in range(2500)]
threadLock = threading.Lock()

def infer_task(sess, ort_inputs):
    stop = False
    while True:
        threadLock.acquire()
        if len(task_list) == 0:
            stop = True
        else:
            task_list.pop()
        threadLock.release()

        if stop:
            break
        else:
            sess.run(None, ort_inputs)



def get_FPS_multithread(context, batch_size, precision, image_list, params, thread_num=8):
    # thread_num = 2
    inputs = get_images(image_list[:batch_size], params)[0:1]
    ort_inputs = {context.get_inputs()[0].name: inputs}
    ## warming up
    for _ in range(100):
        # Run inference.
        context.run(None, ort_inputs)

    threads = []
    for tid in range(thread_num):
        threads.append(threading.Thread(target=infer_task, args=(context, ort_inputs)))

    start = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    time_cost = time.time() - start
    latency = time_cost/(2500)
    FPS = 1./latency

    return FPS   

def infer_task_v1(sess, ort_inputs):
    stop = False
    while True:
        threadLock.acquire()
        if len(task_list) == 0:
            stop = True
        else:
            task_list.pop()
        threadLock.release()

        if stop:
            break
        else:
            sess.run_with_iobinding(ort_inputs)


def get_FPS_multithread_v1(context, batch_size, precision, image_list, params, thread_num=8):
    # thread_num = 2
    inputs = get_images(image_list[:batch_size], params)[0:1]
    ort_inputs = {context.get_inputs()[0].name: inputs}
    # 256 duiqi
    output_names = [output.name for output in context.get_outputs()]
    io_binding = context.io_binding()
    batchsize = 8
    for _ in range(batchsize):
        for key in ort_inputs.keys():
            io_binding.bind_ortvalue_input(key, 
                ort.OrtValue.ortvalue_from_numpy(ort_inputs[key], "macavx_pinned", 0))
        for o_n in output_names:
            io_binding.bind_output(o_n, "macavx_pinned")

    ## warming up
    for _ in range(10):
        # Run inference.
        context.run_with_iobinding(io_binding)

    threads = []
    for tid in range(thread_num):
        threads.append(threading.Thread(target=infer_task_v1, args=(context, io_binding)))

    start = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    time_cost = time.time() - start
    latency = time_cost/(2500)
    FPS = 1./latency * batchsize

    io_binding.clear_binding_inputs()
    io_binding.clear_binding_outputs()

    return FPS    

def split_dataset(image_list, thread_num):
    sub_num = len(image_list)//thread_num
    mod_num = len(image_list)%thread_num
    sub_lists = []
    start_id = 0
    for i in range(thread_num):
        if i < mod_num:
            sub_lists.append(image_list[start_id:start_id+sub_num+1])
            start_id += (sub_num+1)
        else:
            sub_lists.append(image_list[start_id:start_id+sub_num])
            start_id += sub_num
    return sub_lists

def check_accuracy_thread(context, batch_size, precision, params, modelname, test_set):
    post_process = params['post_process']
    # folds_flags = None
    if post_process == "get_mobileface_post":
        # test_set = test_set
        # folds_flags = parseList(test_set)
        batch_size //= 4
    elif post_process == "get_arcface_post":
        batch_size //= 2
    
    batch_num = 0
    predns = []
    for start_idx in trange(0, len(test_set), batch_size):
        batch_num += 1
        # If the number of images in the test set is not divisible by the batch size, the last batch will be smaller.
        # This logic is used for handling that case.
        end_idx = min(start_idx + batch_size, len(test_set))
        effective_batch_size = end_idx - start_idx
        # Do inference for every batch.
        inputs = get_images(test_set[start_idx:start_idx + effective_batch_size], params)
        if post_process == "get_retinaface_post":
            ort_inputs = {context.get_inputs()[0].name: inputs}
            outputs = context.run(None, ort_inputs)     
        else:
            output_list = []
            for i in range(len(inputs)):
                input_ = inputs[i:i+1]
                ort_input = {context.get_inputs()[0].name: input_}
                output = context.run(None, ort_input) 
                output_list.append(output[0])
            outputs = [np.concatenate(output_list,axis=0)]

        preds = get_faces(post_process, outputs, effective_batch_size, params)
        predns.extend(preds)

    return predns

def check_accuracy_multithread(context, batch_size, precision, params, modelname, test_set, thread_num=8):
    print("model init")
    inputs = get_images(test_set[:1], params)[:1]
    ort_inputs = {context.get_inputs()[0].name: inputs}
    context.run(None, ort_inputs)
    
    sub_lists = split_dataset(test_set, thread_num)
    
    threads_pool = ThreadPoolExecutor(max_workers=thread_num)
    future_list = []
    for i, sublist in enumerate(sub_lists):
        future_list.append(threads_pool.submit(check_accuracy_thread,context, batch_size, precision, params, modelname, sublist))
    
    results = [future.result() for future in future_list]
    all_preds = []
    folds_flags = None
    if params['post_process'] == "get_mobileface_post":
        folds_flags = parseList(test_set)
    for result in results:
        all_preds += result

    comute_acc(all_preds, params, modelname, test_set, folds_flags, batch_size, precision)


def check_type_model(modelfile, model_path, params):
    # sup_modeltype = [".pb",".onnx", ".caffemodel",".pd"]
    if modelfile.endswith(".pb"):
        output_model = model_path + "/" + os.path.basename(modelfile).replace(".pb", ".onnx")
        if not os.path.exists(output_model):
            inputs_name = ",".join([name+":0" for name in params['inputs_name'].split(",")])
            outputs_name = ",".join([name+":0" for name in params['outputs_name'].split(",")])
            cmd = "python -m maca_converter --model_path {} --model_type tf-graph --output {} \
                --inputs {} --outputs {} --inputs_as_nchw {}".format(modelfile, output_model, \
                inputs_name, outputs_name, inputs_name )

            os.system(cmd)
            
        return output_model
    elif modelfile.endswith(".caffemodel"):
        output_model = model_path + "/" + os.path.basename(modelfile).replace(".caffemodel", ".onnx")
        if not os.path.exists(output_model):
            caffe_path = os.path.dirname(modelfile)
            cmd = "python -m maca_converter --model_path {} --model_type caffe --output {} "\
                .format(caffe_path, output_model)
            os.system(cmd)
        print("convert caffe to onnx")
        return output_model
    elif modelfile.endswith(".pd"):
        output_model = model_path + "/" + os.path.basename(modelfile).replace(".pd", ".onnx")
        os.system(cmd)
        print("convert paddle to onnx")
        return output_model
    elif modelfile.endswith(".weights"):
        output_model = model_path + "/" + os.path.basename(modelfile).replace(".weights", ".onnx")
        os.system(cmd)
        print("convert darknet to onnx")
        return output_model
    elif modelfile.endswith(".onnx"):
        print("not need to convert, the ori model is onnx")
        return modelfile
    else:
        print("this type of model-{} is not support".format(os.path.basename(modelfile)))
        return modelfile

def check_model(modelfile, modelname, params):
    if ".onnx" in modelfile:
        model_path = modelfile
    else:
        model_path = modelname+"/"+params['model_file'] if params['ori_path'] == "" else \
            params['ori_path']+params['model_file']
            
    return model_path

def get_EP(EP):
    
    provider_list=[]
    if "maca" in EP:
        provider_list+=['MacavxExecutionProvider']
    if "cpu" in EP:
        provider_list+=['CPUExecutionProvider']
    if "cuda" in EP:
        provider_list+=['CUDAExecutionProvider']
    if "trt" in EP:
        provider_list+=['TensorrtExecutionProvider']
    if len(provider_list) == 0:
        provider_list+=['MacavxExecutionProvider']
        
    return provider_list
    
def model_convert(modelname, model_path, precision, params, task):
    if precision == "fp32":
        return model_path
    elif precision == "fp16":
        # fp16_model = model_path.replace(".onnx","_fp16.onnx")
        fp16_model = modelname+"/"+os.path.basename(model_path).replace(".onnx","_fp16.onnx")
        if task == "quanonly":
            print("start convert fp16 model")
            cmd = "python -m maca_converter --model_path {} --model_type onnx --output {} --fp32_to_fp16 1 --simplify 2".format(model_path, fp16_model)
            os.system(cmd)
            return ""
        elif not os.path.exists(fp16_model): 
            print("start convert fp16 model")
            cmd = "python -m maca_converter --model_path {} --model_type onnx --output {} --fp32_to_fp16 1 --simplify 2".format(model_path, fp16_model)
            os.system(cmd)
            # fp16_model = modelname+"/"+os.path.basename(model_path).replace(".onnx","_fp16.onnx")
            if not os.path.exists(fp16_model) or not fp16_model.endswith(".onnx"):
                print("can not convert fp16 onnx model")
                return ""
        return fp16_model
    elif precision == "int8":
        int8_model = modelname+"/"+os.path.basename(model_path).replace(".onnx","_int8_maca.onnx")
        if task == "quanonly":
            from maca_quantizer.maca_quantize_runner import MacaQuantizeRunner
            modify_yaml(modelname+"/quantize.yaml", params, model_path)
            obj = MacaQuantizeRunner(modelname+"/quantize.yaml")
            obj.run()
            print("only quantize int8 model {}".format(int8_model))
            return ""
        elif not os.path.exists(int8_model):
            # if not os.path.exists(model_path.replace(".onnx","_int8_maca.onnx")):
            from maca_quantizer.maca_quantize_runner import MacaQuantizeRunner
            modify_yaml(modelname+"/quantize.yaml", params, model_path)
            obj = MacaQuantizeRunner(modelname+"/quantize.yaml")
            obj.run()
            # else:
                # int8_model = model_path.replace(".onnx","_int8_maca.onnx")
        return int8_model
    else:
        print("can not support {} precision".format(precision))
        return ""






