# vision/classification paddlemodels

## 模型精度支持

|models|fp16|int8|
|:---:|:---:|:---:|
|densenet121|√|√|
|inceptionv4|√|√|
|mobilenetv2|√|√|
|r50_vd_ssld|√|√|
|r101_vd_ssld|√|√|
|squeezenet11|√|√|
|x50_vd_32x4d|√|√|

## 数据集和模型下载
```
# 数据集下载
1. 进入paddlepaddle/vision/classification目录下， 运行./data_prepare.sh

# 部署模型下载
1. 请自行前往driver下载所需模型： https://pub-docstore.metax-tech.com:7001/sharing/8IlMh7DTe
2. 将模型压缩包放置对应模型文件夹内，并解压
```

## demo运行
```
cd paddlepaddle/vision/classification
# ./run.sh ${model_path} ${precision}
./run.sh pd_densenet121_224x224 int8

# demo参数说明
model_path:   存放模型的文件夹，包含模型cfg和部署模型
precision :   模型的推理精度，可选项  fp16和int8
```
 