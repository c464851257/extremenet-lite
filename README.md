### 代码基于https://github.com/xingyizhou/ExtremeNet修改 
## 训练流程
### 1.制作数据集
提供coco数据分割数据集格式转extremenet格式代码，
制作好数据集以后，进入tools文件下运行gen_coco_extreme_points.py
### 2.环境配置
```
# Anaconda
# 创建虚拟环境并安装package
conda create --name ExtremeNet --file conda_packagelist.txt --channel pytorch
# 开启虚拟环境
source activate ExtremeNet
# 退出虚拟环境deactivate
```  
### 3.编译nms
```
cd $ExtremeNet_ROOT/external
make
```
### 4.数据文件夹
需要将制作好的ExtremeNet格式的数据集严格按照以下路径放置
```
${ExtremeNet_ROOT}
|-- data
-- |-- coco
    -- |-- annotations
        |   |-- instances_train2017.json # 训练集标签
        -- images
            |-- train2017 # 训练集图片
```
### 5.参数配置
修改config/ExtremeNet.json文件，根据自己的需求设置 batch_size, max_iter, stepsize, snapshot, chunk_sizes, categories。
【特别注意】代码块第22行chunk_sizes中的所有数加起来等于batch_size。若用2个GPU训练ExtremeNet并且batch_size等于6则chunk_sizes应为[3, 3]。

### 6.模型压缩
代码修改残差块的卷积为深度可分离卷积，若要使用深度可分离卷积，则将models/py_utils/utils.py文件代码第5行设置为True

修改了原代码的边缘聚合部分，并用numba进行加速，大大缩短了推理时间。
增加了深度可分离卷积模块。
