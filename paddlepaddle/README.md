# Paddlepaddle模型库
目前支持paddlepaddle模型有图像分类/检测和OCR，2大类共10个模型，后续会继续增加模型类别和个数


## 测试环境准备
```
1.配置好硬件显卡的driver相应驱动，可按照发布手册自行配置
2.按照发布手册安装onnxruntime-macavx, maca-quantizer, maca-converter内置whl包
3.安装测试运行依赖
	pip install -r requirements.txt
```

## paddle模型测试流程 

### 1.数据集及模型准备
请按相应任务内的readme.md说明下载对应数据集及模型


### 2.模型推理测试
```
# pd_resnet50_vd_ssld_224x224
cd paddlepaddle/vision/classification
./run.sh pd_resnet50_vd_ssld_224x224 int8

# pd_yolov3_608x608
cd paddlepaddle/vision/detection
./run.sh pd_yolov3_608x608 int8

# pd_dbmbv3_736x1280
cd paddlepaddle/ocr
./run.sh pd_dbmbv3_736x1280 int8
```

## 支持模型列表概览

### 图像分类
| 模型| 精度类型 | input-size| CPU/GPU参考指标-top1 | 硬件精度 | 内存 | 硬件利用率 | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|densenet121|int8|224x224|75.7%|75.2%|648.3MB|97%|
|r50_vd_ssld|int8|224x224|83.0%|82.1%|728.3MB|96%|
|r101_vd_ssld|int8|224x224|83.7%|83.4%|824.3MB|95%|
|inceptionv4|int8|299x299|80.8%|80.6%|962.3MB|99%|
|mobilenetv2|int8|224x224|72.2%|71.7%|616.3MB|93%|
|x50_vd_32x4d|int8|224x224|79.6%|79.4%|728.3MB|95%|
|squeezenet11|int8|224x224|60.1%|59.8%|600.3MB|93%|

### 目标检测
| 模型| 精度类型 | input-size| CPU/GPU参考指标-mAP | 硬件精度 | 内存 | 硬件利用率 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|yolov3|int8|608x608|39.1|38.6|1082.3MB|97%|
|ppyolotiny|int8|320x320|20.6|18.6|874.3MB|96%|

### OCR
| 模型| 精度类型 | input-size| CPU/GPU参考指标-Hmean | 硬件精度 | 内存 | 硬件利用率 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|det_mv3_db_v2|int8|736x1280|75.12|74.8|1054.3MB|99%|