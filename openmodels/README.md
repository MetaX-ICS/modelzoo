## 模型库
目前初版modelzoo支持图像分类、检测、分割、人脸检测和图像超分，5种任务共22个模型，后续会继续增加。


## 测试环境准备
1.配置好硬件显卡的driver相应驱动，可按照发布手册自行配置  
2.按照发布手册安装onnxruntime-macavx, maca-quantizer, maca-converter内置whl包   
3.安装测试运行依赖  
```
pip install -r requirements.txt
```
4.数据集准备，可到相应任务的data目录下进行数据集的下载  
```
./*_data_prepare.sh
```
注：部分数据集可能无法直接下载，请参照说明进行下载  
5.模型下载，请自行前往[沐曦模型](https://pub-docstore.metax-tech.com:7001/sharing/8IlMh7DTe)下载所需模型，并将模型压缩包放置对应模型文件夹内，并解压。

## 快速开始

```
# ./run.sh ${mission_path} ${model_path} ${precision}
./run.sh classification models/ox_mobilenetv2_224x224 fp16

# mission_path : 任务文件夹 classification,detection,OCR,face_recognition,segmentation
# model_path   : 模型文件夹 任务文件夹内的目录名相对路径
# precision    : 模型运行精度 fp16, int8
```

## 支持模型列表概览

### 图像分类
| 模型| input-size| CPU-top1精度 | N100-int8精度 | 
|:---:|:---:|:---:|:---:|
|densenet121|224x224|74.5%|73.8%|
|efficientnetb0|224x224|77.7%|73.9%|
|inceptionv3|299x299|77.3%|76.9%|
|inceptionv4|299x299|79.9%|79.8%|
|mobilenetv1|224x224|71.7%|71.3%|
|mobilenetv2|224x224|71.9%|71.3%|
|mobilenetv3|224x224|74.0%|72.6%|
|resnet18|224x224|69.6%|69.3%|
|resnet50|224x224|76.0%|75.4%|
|resnet101|224x224|77.3%|76.7%|
|shufflenetv2|299x299|65.4%|60.4%|
|squeezenet10|224x224|58.0%|57.5%|
|squeezenet11|224x224|58.2%|57.9%|
|vgg16|224x224|71.6%|71.5%|

### 图像检测
| 模型| input-size| CPU-mAP精度 | N100-int8精度 | 
|:---:|:---:|:---:|:---:|
|vgg16_ssd|300x300|77.0%|76.8%|
|mbv1_ssd|300x300|18.3%|17.3%|
|yolov3|416x416|38.3%|36.6%|
|yolov4|608x608|49.4%|46.5%|
|yolov4_leaky|608x608|46.6%|45.1%|

### 图像分割
| 模型| input-size| CPU-mIOU精度 | N100-int8精度 | 
|:---:|:---:|:---:|:---:|
|unet|224x959|99.7%|99.5%|

### 人脸检测
| 模型| input-size| CPU-mAP精度 | N100-int8精度 | 
|:---:|:---:|:---:|:---:|
|retinaface|1000x1000|0.86/0.82/0.66|0.85/0.80/0.64|

### 图像超分
| 模型| input-size| CPU-PSNR精度(2x/4x) | N100-int8精度 | 
|:---:|:---:|:---:|:---:|
|vdsr|512x512|37.25/31.81|37.12/31.76|