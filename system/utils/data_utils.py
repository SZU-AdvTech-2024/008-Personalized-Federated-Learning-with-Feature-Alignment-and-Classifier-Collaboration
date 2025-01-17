import numpy as np
import torch


def read_data(dataset,num_clients,noniid_s, local_size, idx, is_train=True):
    if is_train:
        train_data_dir = 'E:/qianxi/vscode/FedPAC/dataset/'+dataset+"_"+ str(num_clients)+"_"+str(noniid_s)+'_'+str(local_size) +'/train/'

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = 'E:/qianxi/pycharm/FedPAC/dataset/'+dataset +"_"+ str(num_clients)+"_"+str(noniid_s)+'_'+str(local_size)+'/test/'

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data

def read_client_data(dataset,num_clients,noniid_s, local_size,idx, is_train=True):
    if is_train:
        train_data = read_data(dataset,num_clients,noniid_s, local_size, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset,num_clients,noniid_s, local_size,idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
