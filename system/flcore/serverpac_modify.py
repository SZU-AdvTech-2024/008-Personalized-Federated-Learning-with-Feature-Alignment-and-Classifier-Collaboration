import copy
import random
import time
import numpy as np
import torch
from torch import nn
from collections import defaultdict
import os
import csv
# import local file
from flcore.clientpac_modify import clientPAC_M
from utils.data_utils import read_client_data


class FedPAC_M():
    def __init__(self, args):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.noniid_s = args.noniid_s
        self.local_size = args.local_size
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_test_accstd= []
        self.rs_test_aucstd=[]
        self.rs_train_loss = []
        
        self.eval_gap = args.eval_gap

        # self.load_model()
        self.Budget = []
        self.global_protos = [None for _ in range(args.num_classes)]

        self.Vars = []
        self.Hs = []
        self.uploaded_heads = []

        # Create clients
        self.set_clients(clientPAC_M)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")      

        self.loss = nn.CrossEntropyLoss()
        self.decay_rate = 0.5

    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            # 读取客户端数据
            train_data = read_client_data(self.dataset, self.num_clients, self.noniid_s, self.local_size,i, is_train=True)
            test_data = read_client_data(self.dataset,self.num_clients, self.noniid_s, self.local_size,i, is_train=False)
            # 创建客户端对象
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data))
            self.clients.append(client)
    
    def select_clients(self):
        selected_clients = list(np.random.choice(self.clients, self.num_join_clients, replace=False))
        return selected_clients

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.global_protos = self.receive_protos()

            self.send_protos()
            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        self.uploaded_class_weight = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)
            self.uploaded_class_weight.append(client.sample_per_class)

        uploaded_class_weight = defaultdict(list)
        agg_protos_label = defaultdict(list)

        for index, local_protos in enumerate(self.uploaded_protos):  # 每个client的proto
            for label in local_protos.keys():
                agg_protos_label[label].append(local_protos[label])
                uploaded_class_weight[label].append(self.uploaded_class_weight[index][label]/sum(row[label] for row in self.uploaded_class_weight))

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for index, i in enumerate(proto_list):
                    proto += i.data*uploaded_class_weight[label][index]  # 需要权重
                agg_protos_label[label] = proto
            else:
                agg_protos_label[label] = proto_list[0].data

        return agg_protos_label

    def evaluate(self, acc=None, auc=None ,loss=None,accstd=None,aucstd=None, printting=True):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if auc ==None:
            self.rs_test_auc.append(test_auc)
        else:
            auc.append(test_auc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        if accstd == None:
            self.rs_test_accstd.append(np.std(accs))
        else:
            accstd.append(np.std(accs))

        if aucstd == None:
            self.rs_test_aucstd.append(np.std(aucs))
        else:
            aucstd.append(np.std(aucs))

        if printting:
            print("Averaged Train Loss: {:.4f}".format(train_loss))
            print("Averaged Test Accurancy: {:.4f}".format(test_acc))
            print("Averaged Test AUC: {:.4f}".format(test_auc))
            print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
            print("Std Test AUC: {:.4f}".format(np.std(aucs)))
    
    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        # 返回结果
        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_heads = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_heads.append(client.model.head)

    def send_models(self):
        assert (len(self.clients) > 0)
        if self.uploaded_heads != []:
            for client in self.selected_clients:
                start_time = time.time()
                weight = []
                for c in self.selected_clients:
                    w = 0
                    for [label, proto_list] in client.protos.items():
                        out = (c.model.head(proto_list)).to("cpu")
                        ww = client.sample_per_class[label] / client.train_samples
                        onehot = np.zeros(self.num_classes)
                        onehot[label] = 1
                        y_onehot = torch.Tensor(onehot)
                        l = self.loss(out, y_onehot)
                        w += l * ww
                    weight.append(np.exp(-self.decay_rate * w))

                weights = [wei / sum(weight) for wei in weight]

                head = copy.deepcopy(self.uploaded_heads[0])
                for param in head.parameters():
                    param.data.zero_()

                for wei, client_head in zip(weights, self.uploaded_heads):
                    for head_param, client_param in zip(head.parameters(), client_head.parameters()):
                        head_param.data += client_param.data.clone() * wei
                client.set_parameters(head)

                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

            # 如果没有被选择，就直接退化为传输global model
            for client in self.clients:
                if client not in self.selected_clients:
                    start_time = time.time()

                    client.set_parameters(self.global_model)

                    client.send_time_cost['num_rounds'] += 1
                    client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
        else:
            for client in self.clients:
                start_time = time.time()

                client.set_parameters(self.global_model)

                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def save_results(self):
        result_path = 'E:/qianxi/vscode/FedPAC/results/'+self.dataset+'_'+str(self.num_clients)+"_"+str(self.noniid_s)+'_'+str(self.local_size)+'/'
        file_name = str(self.num_clients)+'_'+str(self.join_ratio)
        file_path = result_path + "{}.csv".format(file_name)

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        with open(file_path, 'w', newline='') as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow(['test_acc','test_auc','train_loss','acc_std','auc_std'])
            for i in range(len(self.rs_test_acc)):
                writer.writerow([self.rs_test_acc[i],self.rs_test_auc[i],self.rs_train_loss[i],self.rs_test_accstd[i],self.rs_test_aucstd[i]])

