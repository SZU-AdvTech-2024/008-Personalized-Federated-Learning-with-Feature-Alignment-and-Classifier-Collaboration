源代码详情
一、dataset文件夹：
    1. dataset_utils.py：包含生成数据的公用函数
    2. generate_cifar10.py：下载和划分cifar10数据集
    3. generate_fmnist.py：下载和划分fmnist数据集
二、results文件夹：用于保存实验结果
三、system文件夹：
    1. flcore文件夹：
        (1) clientpac.py：FedPAC客户端算法
        (2) serverpac.py：FedPAC服务器算法
        (3) clientpac_modify.py：修改后的FedPAC客户端算法
        (4) serverpac_modify.py：修改后的FedPAC服务器算法
        (5) models.py：创建模型类
    2. utils文件夹：
        (1) dataset_utils.py：用于读取数据集
    3. config.py：用于设置参数
    4. main.py：运行算法