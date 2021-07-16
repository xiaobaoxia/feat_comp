### Data
预训练参数地址 https://drive.google.com/drive/folders/1QL9lpEeTgzJMCEZ2m-9gOxGr6TChB2PU

需要下载预训练参数到```../model/```文件夹，使用model QP1-7

需要下载COCO2017数据集到配置中的路径 ```../../../coco_dataset```
可以只下载 trainval 和 annotation 数据集目录结构如下
![image](https://user-images.githubusercontent.com/33535066/125921637-6ca07c80-1d71-4559-95d6-cc834386fbf8.png)

需要安装的包在文件```../requirement.txt```中

### Command

两张卡训练网络 不加载预训练参数

```shell
python train.py --batchsize 16 --gpu "0,1" --gpu_count 2 --num_workers 16 --qp 7 --load_weights 0 --coco_root ../../../coco_dataset
```

以下是测试使用的训练参数

如果想要提升训练速度，需要增加GPU数量和batchsize，并且gpu_count需要同时修改为使用的GPU数量

通过QP参数加载不同的预训练参数和 lambda

num_workers是加载训练数据使用的进程数量，增加workers也可以在一定程度上增加训练速度，但是会消耗内存，CPU，GPU。所以num_workers不能太高，需要权衡

learning_rate 目前看来设置为默认值效果最好

训练的模型存放在```train```中
