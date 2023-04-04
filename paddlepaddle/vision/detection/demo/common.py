#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import time
import json
import os
import subprocess

import threading
from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort

import cv2
import numpy as np
import onnx
from tqdm import trange
import yaml

from utils import  get_params, preprocess_test, get_images, get_bboxes, comute_mAP_coco, comute_mAP_voc

import logging


def comute_mAP(preds, params, save_path, iters, batchsize=1, precision="fp16"):
    dataset_type = "COCO" if "dataset" not in params.keys() else params['dataset']
    if dataset_type == "COCO":
        mAP, mAP50 = comute_mAP_coco(preds, params, save_path, iters)
        print("detec_{}_bs{}_prec{} result: mAP : {:.2f}%, mAP50 : {:.2f}%".format(save_path.replace("/", "_") ,batchsize,precision, mAP*100, mAP50*100))
        # print("detec_{}_bs{}_prec{} result: mAP50 : {:.2f}%".format(save_path.replace("/", "_") ,batchsize, precision, mAP50*100))
        logging.info("detec_{}_bs{}_prec{} result: mAP : {:.2f}%, mAP50 : {:.2f}%".format(save_path.replace("/", "_") ,batchsize,precision, mAP*100, mAP50*100))
        # logging.info("detec_{}_bs{}_prec{} result: mAP50 : {:.2f}%".format(save_path.replace("/", "_") ,batchsize, precision, mAP50*100))
    elif dataset_type == "VOC":
        mAP50 = comute_mAP_voc(preds, params, save_path, iters)
        print("detec_{}_bs{}_prec{} result: mAP50 : {:.2f}%".format(save_path.replace("/", "_") ,batchsize, precision, mAP50*100))
        logging.info("detec_{}_bs{}_prec{} result: mAP50 : {:.2f}%".format(save_path.replace("/", "_") ,batchsize, precision, mAP50*100))
    else:
        pass

def modify_yaml(yaml_file, params, modelfile, export_type="native"):
    # with open(yaml_file, "r") as f:
    #     content = yaml.load(f.read())

    model_path = os.path.split(yaml_file)[0]

    content = {}

    content["dataset"] = {}
    content["dataset"]["batch_size"] = 8 
    if params["post_process"] =="get_resnet34_ssd_bboxes":
        content["dataset"]["batch_size"] = 1
    elif params["post_process"] =="get_effc_bboxes":
        content["dataset"]["batch_size"] = 4
    else:
        pass
    content["dataset"]["calib_dir"] = "./data/val2017"
    content["dataset"]["calib_num"] = 200 if params["post_process"] !="get_effc_bboxes" else 50
    content["dataset"]["preprocessing"] = {}
    content["dataset"]["preprocessing"]["enable"] = True
    content["dataset"]["preprocessing"]["type"] = "preprocess_v1"
    content["dataset"]["preprocessing"]["attributes"] = {}
    mean = [float(value) for value in params['mean'].split(",")]
    std = [float(value) for value in params['std'].split(",")] if "std" in params.keys() \
        else [1./float(value) for value in params['scale'].split(",")]
    content["dataset"]["preprocessing"]["attributes"]["mean"] = [value if value>1 else value*255. for value in mean] if len(mean) != 1 \
        else [value if value>1 else value*255. for value in mean]*3
    content["dataset"]["preprocessing"]["attributes"]["std"] = [value if value>1 else value*255. for value in std] if len(std) != 1 \
        else [value if value>=1 else value*255. for value in std]*3
    if params["post_process"]=="get_resnet34_ssd_bboxes" or params["post_process"]=="get_yolov3_mmdeploy_postprocess":
         content["dataset"]["preprocessing"]["attributes"]["std"] = [ value*255. for value in std]*3
    content["dataset"]["preprocessing"]["attributes"]["isreverses"] = True if "reverse_channel" in params.keys() else False
    content["dataset"]["preprocessing"]["attributes"]["resize"] = {}
    size = [int(value) for value in params['input_size'].split(",")]
    content["dataset"]["preprocessing"]["attributes"]["resize"]["keep_ratio"] = False if params["post_process"]!="get_yolox_bboxes" else True
    content["dataset"]["preprocessing"]["attributes"]["resize"]["to"] = size
    content["dataset"]["preprocessing"]["attributes"]["resize"]["pad_value"] = 0 if params["post_process"]!="get_yolox_bboxes" else 114

    content["import_model"] = modelfile
    content["export_model"] = model_path + "/" +  os.path.basename(modelfile).replace(".onnx", "_int8_maca.onnx")
    content["export_type"] = "maca"

    if params["post_process"]=="get_rfb_bboxes":
        content["quant_algorithm"] = "minmax"
    if params["post_process"]=="get_mobilenetv1_ssd_bboxes_v1" or params["post_process"]=="get_vgg16_ssd_bboxes_v1":
        content["force_advance_quant"] = True

    with open(yaml_file, "w") as fw:
        yaml.dump(content,fw)
    # common.write_yml(content, yaml_file)

    print("yaml file is :"+yaml_file)
    print("+++++++++++++++++++++ modify_yaml is Success ++++++++++++++")


task_list = [i for i in range(1000)]
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
    inputs = get_images(image_list[:batch_size], params)
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
    latency = time_cost/(1000)
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

dla_stop = False
def get_dla_util(dla_utils):
    while True:
        if dla_stop:
            break
        dla_util = get_cmd_output_with_list("mx-smi --show-usage|awk 'NR==9'")
        dla_utils.append(dla_util)
        time.sleep(0.1)
        

def get_FPS_multithread_v1(context, batch_size, precision, image_list, params, thread_num=8):
    # thread_num = 2
    inputs = get_images(image_list[:batch_size], params)
    ort_inputs = {context.get_inputs()[0].name: inputs}
    # 256 duiqi
    output_names = [output.name for output in context.get_outputs()]
    io_binding = context.io_binding()
    batchsize = 1 if "ppyolo_tiny" not in params["model_file"] else 8 
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
    print("warming up done")

    threads = []
    for tid in range(thread_num):
        threads.append(threading.Thread(target=infer_task_v1, args=(context, io_binding)))
        
    test_num = 1000 if "ppyolo_tiny" not in params["model_file"] else 10000
    # print(params["model_file"])
    if precision == "int8":
        test_num = max(5000, test_num)
    global task_list
    task_list = [i for i in range(test_num)]

    dla_utils = ['0']
    utils_thread = threading.Thread(target=get_dla_util, args=(dla_utils,))
    
    start = time.time()
    for thread in threads:
        thread.start()
    utils_thread.start()
    for thread in threads:
        thread.join()
    global dla_stop
    dla_stop =True
    utils_thread.join()
    
    time_cost = time.time() - start
    latency = time_cost/(test_num)
    FPS = 1./latency * batchsize

    io_binding.clear_binding_inputs()
    io_binding.clear_binding_outputs()

    return FPS, dla_utils

def split_dataset(image_list, thread_num):
    sub_num = len(image_list)//thread_num
    mod_num = len(image_list)%thread_num
    sub_lists = []
    start_id = 0
    si_list = []
    for i in range(thread_num):
        if i < mod_num:
            sub_lists.append(image_list[start_id:start_id+sub_num+1])
            si_list.append(start_id)
            start_id += (sub_num+1)
        else:
            sub_lists.append(image_list[start_id:start_id+sub_num])
            si_list.append(start_id)
            start_id += sub_num
    return sub_lists, si_list

def check_accuracy_thread(context, batch_size, precision, params, modelname, test_set, si):
    post_process = params['post_process']
    # print("thread {} start".format(si))
    batch_num = 0
    predns = []
    for start_idx in trange(0, len(test_set), batch_size):
        batch_num += 1
        end_idx = min(start_idx + batch_size, len(test_set))
        effective_batch_size = end_idx - start_idx
        # Do inference for every batch.
        inputs = get_images(test_set[start_idx:start_idx + effective_batch_size], params)
        ort_inputs = {context.get_inputs()[0].name: inputs}

        outputs = context.run(None, ort_inputs) 

        preds = get_bboxes(post_process, outputs, effective_batch_size, params, start_idx+si)
        predns.extend(preds)
    # print("thread {} done".format(si))
    return predns

def check_accuracy_multithread(context, batch_size, precision, params, modelname, test_set, thread_num=8):
    print("model init")
    inputs = get_images(test_set[:1], params)
    ort_inputs = {context.get_inputs()[0].name: inputs}
    context.run(None, ort_inputs)
    
    sub_lists, si_list = split_dataset(test_set, thread_num)
    
    threads_pool = ThreadPoolExecutor(max_workers=thread_num)
    future_list = []
    for i, (sublist, si) in enumerate(zip(sub_lists,si_list)):
        future_list.append(threads_pool.submit(check_accuracy_thread,context, batch_size, precision, params, modelname, sublist, si))
    
    all_preds = []
    for future in future_list:
        all_preds += future.result()
    print("all threads done")
    comute_mAP(all_preds, params, modelname, len(test_set), batch_size, precision)


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
        modelfile = modelfile.replace(".pd", "")
        output_model = model_path + "/" + os.path.basename(modelfile) + ".onnx"
        if not os.path.exists(output_model):
            cmd = "python -m maca_converter --model_path {} --model_type paddle --output {} \
            --simplify 2 >/dev/null".format(modelfile, output_model )
            os.system(cmd)
            if not os.path.exists(output_model):
                print("can not convert paddle to  onnx model")
                return ""
            sup_swish = "python -m maca_converter --model_path {} --model_type onnx --output {} \
            --support_swish 1 --gemm_optimization 1 --matmul_to_gemm 1 >/dev/null".format(output_model, output_model )
            os.system(sup_swish)
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
        model_path = "./" + modelname+"/"+params['model_file'] if params['ori_path'] == "" else \
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
            cmd = "python -m maca_converter --model_path {} --model_type onnx --output {} --fp32_to_fp16 1 --simplify 0 >/dev/null".format(model_path, fp16_model)
            os.system(cmd)
            return ""
        elif not os.path.exists(fp16_model): 
            print("start convert fp16 model")
            cmd = "python -m maca_converter --model_path {} --model_type onnx --output {} --fp32_to_fp16 1 --simplify 0 >/dev/null".format(model_path, fp16_model)
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
            if not os.path.exists(modelname+"/quantize.yaml"):
                modify_yaml(modelname+"/quantize.yaml", params, model_path)
            obj = MacaQuantizeRunner(modelname+"/quantize.yaml")
            obj.run()
            print("only quantize int8 model {}".format(int8_model))
            return ""
        elif not os.path.exists(int8_model):
            # if not os.path.exists(model_path.replace(".onnx","_int8_maca.onnx")):
            from maca_quantizer.maca_quantize_runner import MacaQuantizeRunner
            if not os.path.exists(modelname+"/quantize.yaml"):
                modify_yaml(modelname+"/quantize.yaml", params, model_path)
            obj = MacaQuantizeRunner(modelname+"/quantize.yaml")
            obj.run()
            # else:
                # int8_model = model_path.replace(".onnx","_int8_maca.onnx")
        return int8_model
    else:
        print("can not support {} precision".format(precision))
        return ""

def get_cmd_output_with_list(cmd):
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0].decode().split('\n')
    output = [i.strip() for i in output if i != ''][0]
    
    output = output.split(":")[-1].strip()
    
    return output 
