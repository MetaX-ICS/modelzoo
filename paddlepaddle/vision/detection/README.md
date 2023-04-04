# vision/detection paddlemodels

## 模型精度支持

|models|fp16|int8|
|:---:|:---:|:---:|
|yolov3|√|√|
|ppyolotiny|√|√|

## 数据集和模型下载
```
# 数据集下载
进入paddlepaddle/vision/detection目录下， 运行./data_prepare.sh

# 部署模型下载
1. 请自行前往driver下载所需模型： https://pub-docstore.metax-tech.com:7001/sharing/8IlMh7DTe
2. 将模型压缩包放置对应模型文件夹内，并解压
```

## demo运行

```
cd paddlepaddle/vision/detection
# /run.sh ${model_path} ${precision}
./run.sh pd_yolov3_608x608 int8

# demo参数说明
model_path:   存放模型的文件夹，包含模型cfg和部署模型
precision :   模型的推理精度，可选项  fp16和int8
```
 