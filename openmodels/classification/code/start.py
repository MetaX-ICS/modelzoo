#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import json
from common import get_params, check_type_model, check_model

# UFF_Engine = "./code/TensorRT/sample-uff.py"
ONNX_Engine = "./code/OnnxRT/sample_onnx_multithreads.py"
# ONNX_Engine = "./code/OnnxRT/sample_onnx.py"
# PPQ_Engine = "./code/OnnxRT/sample_ppq.py"


def main(modelname, batchsize, precision, task="normal", modelfile="./", EP="maca", thr_num="16"):
    params = get_params(modelname)
    model_path = check_model(modelfile, modelname, params)

    cmd = "" 
    model_path = check_type_model(model_path, modelname, params)
    if model_path != "":
        cmd += "python " + ONNX_Engine + ' ' + modelname + ' ' + batchsize + ' ' + precision + ' ' \
                + task + ' ' + model_path + ' ' + EP + ' ' +  thr_num
        
        response = os.system(cmd)
    

if __name__ == "__main__":
    modelname = sys.argv[1]
    batchsize = sys.argv[2]
    precision = sys.argv[3]
    task = sys.argv[4]
    model_path = sys.argv[5] if len(sys.argv) > 5 else "./"
    EP = sys.argv[6] if len(sys.argv) > 6 else "maca"
    thr_num = sys.argv[7] if len(sys.argv) > 7 else "16"
    main(modelname, batchsize, precision, task, model_path, EP, thr_num)

