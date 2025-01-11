import os
import ujson
import numpy as np


# 检查是否已经生成了指定配置的数据集，以避免重复生成
def check(config_path, train_path, test_path, num_clients, num_classes,noniid_s, local_size, batch_size, niid=False, 
        balance=True, partition=None):
    # check existing dataset
    # 检查配置文件是否存在
    if os.path.exists(config_path):
        # 打开配置文件以读取其中的配置
        with open(config_path, 'r') as f:
            # 加载配置中的JSON数据，将其转换为字典
            config = ujson.load(f)
        # 检查配置文件中参数是否和当前函数参数一致，若一致则已含有此数据集，返回True
        if config['num_clients'] == num_clients and \
            config['num_classes'] == num_classes and \
            config['noniid_s'] == noniid_s and \
            config['local_size'] == local_size and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition and \
            config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    # 若训练集目录不存在，创建目录
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # 若测试集目录不存在，创建目录
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

# 保存生成的训练数据、测试数据以及相关的配置信息
def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
                num_classes, noniid_s, local_size, batch_size, niid=False, balance=True, partition=None):
    # 配置信息
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'noniid_s':noniid_s,
        'local_size':local_size,
        'batch_size': batch_size, 
    }

    # gc.collect()
    print("Saving to disk.\n")

    # 保存数据
    for idx in range(len(train_data)):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_data[idx])
    for idx in range(len(test_data)):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_data[idx])
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
