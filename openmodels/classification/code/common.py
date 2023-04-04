#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import time
import json
import os
import onnx

import threading
from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort

import cv2
import numpy as np
from preprocess import get_images
import yaml

from tqdm import trange

import logging
# logging.basicConfig(level=logging.INFO #设置日志输出格式
#                     ,filename="./LOGGING_ORT.log" #log日志输出的文件位置和文件名
#                     # ,filemode="w" #文件的写入格式，w为重新写入文件，默认是追加
#                     ,format="%(asctime)s - %(name)s - %(levelname)-9s: %(message)s" #日志输出的格式  - %(filename)-8s  - %(lineno)s line - 
#                     # -8表示占位符，让输出左对齐，输出长度都为8位
#                     ,datefmt="%Y-%m-%d %H:%M:%S" #时间输出的格式
#                     )

Jsonfile = "./%s/config.json"


def get_params(modelname):
    with open(Jsonfile%(modelname), 'r') as f:
        param_all = json.load(f)
        params = param_all["model_config"]
    return params

def preprocess_test(path):
    image_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            image_list.append(line.strip('\n').split())

    return image_list


def check_accuracy_v1(context, batch_size, precision, cls_num, image_list, params):

    top1_correct = 0
    top5_correct = 0
    num_total = 0
    batch_num = 0
    for start_idx in trange(0, len(image_list), batch_size):
        batch_num += 1
        end_idx = min(start_idx + batch_size, len(image_list))
        effective_batch_size = end_idx - start_idx
        # Do inference for every batch.
        inputs, labels = get_images(image_list[start_idx:start_idx + effective_batch_size], params)
        ort_inputs = {context.get_inputs()[0].name: inputs}

        [output] = context.run(None, ort_inputs)

        preds_top1 = np.argmax(output.reshape(-1, cls_num)[0:effective_batch_size], axis=1)
        preds_top5 = np.argsort(-output.reshape(-1, cls_num)[0:effective_batch_size], axis=1)[:, :5]
        # labels = test_labels[start_idx:start_idx + effective_batch_size]

        num_total += effective_batch_size
        top1_correct += np.count_nonzero(np.equal(preds_top1, labels))

        for ix, in_ in enumerate(preds_top5):
            if (in_ == labels[ix]).any():
                top5_correct += 1

    percent_top1_correct = 100 * top1_correct / float(num_total)
    percent_top5_correct = 100 * top5_correct / float(num_total)

    return percent_top1_correct, percent_top5_correct

def modify_yaml(yaml_file, params, modelfile, export_type="native"):
    # with open(yaml_file, "r") as f:
    #     content = yaml.load(f.read())

    model_path = os.path.split(yaml_file)[0]#yaml_file.split("/")[0]

    content = {}

    content["dataset"] = {}
    content["dataset"]["batch_size"] = 16
    content["dataset"]["calib_dir"] = "./data/val_old"
    content["dataset"]["calib_num"] = 200
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
    content["dataset"]["preprocessing"]["attributes"]["isreverses"] = True if "reverse_channel" in params.keys() else False
    content["dataset"]["preprocessing"]["attributes"]["resize"] = {}
    size = [int(value) for value in params['input_size'].split(",")]
    if size[1] <256:
        content["dataset"]["preprocessing"]["attributes"]["resize"]["keep_ratio"] = False #True
        content["dataset"]["preprocessing"]["attributes"]["resize"]["centercrop"] = size[1:3]
        content["dataset"]["preprocessing"]["attributes"]["resize"]["to"] = [3,256,256]
    else:
        content["dataset"]["preprocessing"]["attributes"]["resize"]["keep_ratio"] = False
        content["dataset"]["preprocessing"]["attributes"]["resize"]["to"] = size

    content["import_model"] = modelfile
    content["export_model"] = model_path + "/" + os.path.basename(modelfile).replace(".onnx", "_int8_maca.onnx")
    content["export_type"] = "maca"
    content["optimize_output_level"] = 2
    if os.path.basename(modelfile).startswith("mobilenet_v3") or os.path.basename(modelfile).startswith("densenet121") \
        or os.path.basename(modelfile).startswith("mobilenet_v1") or os.path.basename(modelfile).startswith("ResNeXt50_vd"):
        content["force_advance_quant"] = True
    # if os.path.basename(modelfile).startswith("mobilenet_v3"):
    #     content["dataset"]["calib_dir"] = params["label_list"]
    #     content["dataset"]["preprocessing"]["enable"] = False

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
    inputs, _ = get_images(image_list[:batch_size], params)
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
    inputs, _ = get_images(image_list[:1], params)
    ort_inputs = {context.get_inputs()[0].name: inputs}
    # 256 duiqi
    output_names = [output.name for output in context.get_outputs()]
    io_binding = context.io_binding()
    # batchsize = 8
    for _ in range(batch_size):
        for key in ort_inputs.keys():
            io_binding.bind_ortvalue_input(key, 
                ort.OrtValue.ortvalue_from_numpy(ort_inputs[key], "macavx_pinned", 0))
        for o_n in output_names:
            io_binding.bind_output(o_n, "macavx_pinned")

    ## warming up
    for _ in range(100):
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
    FPS = 1./latency * batch_size

    io_binding.clear_binding_inputs()
    io_binding.clear_binding_outputs()

    return FPS    

def check_accuracy_multithread(sess, batch_size, precision, cls_num, image_list, params, thread_num=8):
    print("model init")
    inputs, _ = get_images(image_list[:1], params)
    ort_inputs = {sess.get_inputs()[0].name: inputs}
    sess.run(None, ort_inputs)
    
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

    # [image_list[i:i+sub_num] for i in range(0,len(image_list),sub_num)]
    
    threads_pool = ThreadPoolExecutor(max_workers=thread_num)
    future_list = []
    for i, sublist in enumerate(sub_lists):
        future_list.append(threads_pool.submit(check_accuracy_v1,sess, batch_size, precision, cls_num, sublist, params))
        
    result_list = [future.result() for future in future_list]

    top1_res = sum([res[0] for res in result_list])/len(result_list)
    top5_res = sum([res[1] for res in result_list])/len(result_list)

    # print("class_{}_bs{}_prec{}_top1_multi Acc: {:.2f}%".format(modelname.replace("/", "_"), batch_size, precision, top1_res))
    # print("class_{}_bs{}_prec{}_top5_multi Acc: {:.2f}%".format(modelname.replace("/", "_"), batch_size, precision, top5_res))
    return top1_res, top5_res


def check_type_model(modelfile, model_path, params):
    # sup_modeltype = [".pb",".onnx", ".caffemodel",".pd"]
    model_type = params["model_type"] if "model_type" in params.keys() else "onnx"
    if model_type == "tf":
        output_model = model_path + "/" + os.path.basename(modelfile).replace(".pb", ".onnx")
        if not os.path.exists(output_model):
            inputs_name = ",".join([name+":0" for name in params['inputs_name'].split(",")])
            outputs_name = ",".join([name+":0" for name in params['outputs_name'].split(",")])
            cmd = "python -m maca_converter --model_path {} --model_type tf-graph --output {} \
                --inputs {} --outputs {} --inputs_as_nchw {}".format(modelfile, output_model, \
                inputs_name, outputs_name, inputs_name )
            os.system(cmd)
        return output_model
    elif model_type == "caffe":
        output_model = model_path + "/" + os.path.basename(modelfile).replace(".caffemodel", ".onnx")
        if not os.path.exists(output_model):
            caffe_path = os.path.dirname(modelfile)
            cmd = "python -m maca_converter --model_path {} --model_type caffe --output {} "\
                .format(caffe_path, output_model)
            os.system(cmd)
        print("convert caffe to onnx")
        return output_model
    elif model_type == "paddle":
        modelfile = modelfile.replace(".pd", "")
        output_model = model_path + "/" + os.path.basename(modelfile) + ".onnx"
        if not os.path.exists(output_model):
            cmd = "python -m maca_converter --model_path {} --model_type paddle --output {} \
            --simplify 2".format(modelfile, output_model )
            os.system(cmd)
            if not os.path.exists(output_model):
                print("can not convert paddle to  onnx model")
                return ""
            sup_swish = "python -m maca_converter --model_path {} --model_type onnx --output {} \
            --support_swish 1 --gemm_optimization 1 --matmul_to_gemm 1".format(output_model, output_model )
            os.system(sup_swish)
            print("convert paddle to onnx")
        return output_model
    elif model_type == "darknet":
        output_model = model_path + "/" + os.path.basename(modelfile).replace(".weights", ".onnx")
        os.system(cmd)
        print("convert darknet to onnx")
        return output_model
    elif model_type == "onnx":
        print("not need to convert, the ori model is onnx")
        return modelfile
    elif model_type == "torch":
        modelfile = modelfile.replace(".pth", "").replace(".pt", "")
        output_model = model_path + "/" + os.path.basename(modelfile) + ".onnx"
        if not os.path.exists(output_model):
            cmd = "python -m maca_converter --model_path {}.pth --model_type pytorch --output {} \
                    --model_class_name torchvision.models.{} --model_weights_file {}.pth \
                    --input_shape [1,{}] --simplify 0"\
                    .format(modelfile, output_model, os.path.basename(modelfile),modelfile,params["input_size"])
            os.system(cmd)
            if not os.path.exists(output_model):
                print("can not convert pth to  onnx model")
                return ""
            print("convert torch to onnx")
        return output_model   
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
        sim = 2 #if "pd_" not in fp16_model else 2
        cmd = "python -m maca_converter --model_path {} --model_type onnx --output {} --fp32_to_fp16 1 --simplify {}".format(model_path, fp16_model, sim)
        if task == "quanonly":
            print("start convert fp16 model")
            os.system(cmd)
            return ""
        elif not os.path.exists(fp16_model): 
            print("start convert fp16 model")
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

