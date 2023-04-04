# OCR paddlemodels

## 模型精度支持

|models|fp16|int8|
|:---:|:---:|:---:|
|db_mbv3|×|√|

## 数据集和模型下载
```
# 数据集下载
1. 请自行前往百度云盘下载: https://pan.baidu.com/s/1NESrr21RFsaT4xjdhBNEFQ
提取码： 1111  
重名名为ICDAR2015_dataset.zip
2. 将下载到的 ICDAR2015_dataset.zip 放置到model/ocr/data文件夹内
3. 进入paddlepaddle/ocr目录下， 运行./data_prepare.sh

# 部署模型下载 
1. 请自行前往driver下载所需模型： https://pub-docstore.metax-tech.com:7001/sharing/8IlMh7DTe
2. 将模型压缩包放置对应模型文件夹内，并解压
```

## demo运行
```
cd paddlepaddle/ocr
# ./run.sh ${model_path} ${precision}
./run.sh pd_dbmbv3_736x1280 int8

# demo参数说明
model_path:   存放模型的文件夹，包含模型cfg和部署模型
precision :   模型的推理精度，可选项  fp16和int8
```
 