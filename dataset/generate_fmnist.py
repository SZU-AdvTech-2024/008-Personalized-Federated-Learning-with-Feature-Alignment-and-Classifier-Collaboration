import numpy as np
import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
from dataset_utils import check, save_file

random.seed(1)
np.random.seed(1)
batch_size = 50
num_clients = 100
num_classes = 10
local_size = 600
noniid_s = 20
dir_path = "fmnist_"+str(num_clients)+"_"+str(noniid_s)+'_'+str(local_size)+"/"

# Allocate data to users
def generate_fmnist(niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes,noniid_s, local_size, batch_size ,niid, balance, partition):
        return

    # Get FashionMNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.FashionMNIST(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    train_data = fmnist_noniid_s(trainset, num_clients)
    test_data = fmnist_noniid_s (testset, num_clients)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
         niid, balance, partition)

def fmnist_noniid_s(dataset, num_users, noniid_s=20, local_size=600):
    np.random.seed(2022)
    s = noniid_s/100
    num_per_user = local_size
    num_classes = len(np.unique(dataset.targets))
    noniid_labels_list = [[0,1,2], [2,3,4], [4,5,6], [6,7,8], [8,9,0]]
    # -------------------------------------------------------
    # divide the dataset
    num_imgs_iid = int(num_per_user*s)              # iid样本数量
    num_imgs_noniid = num_per_user - num_imgs_iid   # non-iid样本数量

    after_dataset = {i:{'x':[],'y':[]} for i in range(num_users)}

    num_samples = len(dataset)
    num_per_label_total = int(num_samples/num_classes)  # 每个类别有多少数据

    labels1 = np.array(dataset.targets)
    data1 = np.array(dataset.data)
    idxs1 = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs1, labels1))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    label_used = [0 for i in range(num_classes)]
    iid_per_label = int(num_imgs_iid/num_classes)

    np.random.seed(2022)
    for i in range(num_users):
        # allocate iid idxs
        label_cnt = 0
        for y in label_list:
            label_cnt = label_cnt + 1
            iid_num = iid_per_label
            start = y*num_per_label_total+label_used[y]
            if (label_used[y]+iid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            add_idxs = idxs[start:start+iid_num]
            after_dataset[i]['y'] = np.concatenate((after_dataset[i]['y'],[y for _ in range(len(add_idxs))]),axis=0)
            after_dataset[i]['x'] += [data1[add_idxs[a]] for a in range(len(add_idxs))]
            label_used[y] = label_used[y] + iid_num
        # allocate noniid idxs
        rand_label = noniid_labels_list[i%5]
        noniid_labels = len(rand_label)
        noniid_per_num = int(num_imgs_noniid/noniid_labels)
        noniid_per_num_last = num_imgs_noniid - noniid_per_num*(noniid_labels-1)
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            noniid_num = noniid_per_num
            start = y*num_per_label_total+label_used[y]
            if label_cnt == noniid_labels:
                noniid_num = noniid_per_num_last
            if (label_used[y]+noniid_num)>num_per_label_total:
                start = y*num_per_label_total
                label_used[y] = 0
            add_idxs = idxs[start:start+noniid_num]
            after_dataset[i]['y'] = np.concatenate((after_dataset[i]['y'],[y for _ in range(len(add_idxs))]),axis=0)
            after_dataset[i]['x'] += [data1[add_idxs[a]] for a in range(len(add_idxs))]
            label_used[y] = label_used[y] + noniid_num

    return after_dataset

if __name__ == "__main__":
    # niid = True if sys.argv[1] == "noniid" else False
    # balance = True if sys.argv[2] == "balance" else False
    # partition = sys.argv[3] if sys.argv[3] != "-" else None

    niid = False
    balance = False
    partition = None

    generate_fmnist(niid, balance, partition)