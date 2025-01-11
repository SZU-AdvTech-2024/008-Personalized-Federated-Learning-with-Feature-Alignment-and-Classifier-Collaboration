from collections import defaultdict
import time
import numpy as np
import torch
import cvxpy as cvx
import copy
import os
import csv
import torch.nn as nn
# import local file
from flcore.clientpac import ClientPAC
from utils.data_utils import read_client_data


class FedPAC( ):
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
        self.set_clients(ClientPAC)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

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
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            self.Vars = []
            self.Hs = []
            for client in self.selected_clients:
                self.Vars.append(client.V)
                self.Hs.append(client.h)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_protos()
            self.global_protos = proto_aggregation(self.uploaded_protos)
            self.send_protos()

            self.receive_models()
            self.aggregate_parameters()

            self.aggregate_and_send_heads()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
    
    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def send_models(self):
        # 确保当前服务器对象中存在至少一个客户端
        assert (len(self.clients) > 0)
        # 遍历所有客户端并发送全局模型
        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            # 更新发送时间统计
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)

    def evaluate(self, acc=None, auc=None ,loss=None,accstd=None,aucstd=None):
        stats = self.test_metrics() #ids, num_samples, tot_correct, tot_auc
        stats_train = self.train_metrics()  #ids, num_samples, losses

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
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
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_heads = []
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.base)
            self.uploaded_heads.append(client.model.head)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_and_send_heads(self):
        head_weights = solve_quadratic(len(self.uploaded_ids), self.Vars, self.Hs)

        for idx, cid in enumerate(self.uploaded_ids):
            print('(Client {}) Weights of Classifier Head'.format(cid))
            print(head_weights[idx],'\n')

            if head_weights[idx] is not None:
                new_head = self.add_heads(head_weights[idx])
            else:
                new_head = self.uploaded_heads[cid]

            self.clients[cid].set_head(new_head)

    def add_heads(self, weights):
        new_head = copy.deepcopy(self.uploaded_heads[0])
        for param in new_head.parameters():
            param.data.zero_()
                    
        for w, head in zip(weights, self.uploaded_heads):
            for server_param, client_param in zip(new_head.parameters(), head.parameters()):
                server_param.data += client_param.data.clone() * w
        return new_head

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        # 从上传的模型列表中深拷贝第一个模型，作为全局模型的初始值
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        # 将全局模型参数置零
        for param in self.global_model.parameters():
            param.data.zero_()

        # 累加客户端模型的参数
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        # 遍历全局模型和客户端模型的参数
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            # 累加加权参数
            server_param.data += client_param.data.clone() * w

    def save_results(self):
        result_path = 'E:/qianxi/pycharm/FedPAC/results/'+self.dataset+'_'+str(self.num_clients)+"_"+str(self.noniid_s)+'_'+str(self.local_size)+'/'
        file_name = str(self.num_clients)+'_'+str(self.join_ratio)
        file_path = result_path + "{}.csv".format(file_name)

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        with open(file_path, 'w', newline='') as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow(['test_acc','test_auc','train_loss','acc_std','auc_std'])
            for i in range(len(self.rs_test_acc)):
                writer.writerow([self.rs_test_acc[i],self.rs_test_auc[i],self.rs_train_loss[i],self.rs_test_accstd[i],self.rs_test_aucstd[i]])

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label


# https://github.com/JianXu95/FedPAC/blob/main/tools.py#L94
def solve_quadratic(num_users, Vars, Hs):
    device = Hs[0][0].device
    num_cls = Hs[0].shape[0] # number of classes
    d = Hs[0].shape[1] # dimension of feature representation
    avg_weight = []
    for i in range(num_users):
        # ---------------------------------------------------------------------------
        # variance ter
        v = torch.tensor(Vars, device=device)
        # ---------------------------------------------------------------------------
        # bias term
        h_ref = Hs[i]
        dist = torch.zeros((num_users, num_users), device=device)
        for j1, j2 in pairwise(tuple(range(num_users))):
            h_j1 = Hs[j1]
            h_j2 = Hs[j2]
            h = torch.zeros((d, d), device=device)
            for k in range(num_cls):
                h += torch.mm((h_ref[k]-h_j1[k]).reshape(d,1), (h_ref[k]-h_j2[k]).reshape(1,d))
            dj12 = torch.trace(h)
            dist[j1][j2] = dj12
            dist[j2][j1] = dj12

        # QP solver
        p_matrix = torch.diag(v) + dist
        p_matrix = p_matrix.cpu().numpy()  # coefficient for QP problem
        evals, evecs = torch.linalg.eig(torch.tensor(p_matrix))
        
        # for numerical stablity
        p_matrix_new = 0
        p_matrix_new = 0
        for ii in range(num_users):
            if evals[ii].real >= 0.01:
                p_matrix_new += evals[ii].real*torch.mm(evecs[:,ii].reshape(num_users,1), evecs[:,ii].reshape(1, num_users))
        p_matrix = p_matrix_new.numpy() if not np.all(np.linalg.eigvals(p_matrix)>=0.0) else p_matrix
        
        # solve QP
        alpha = 0
        eps = 1e-3
        if np.all(np.linalg.eigvals(p_matrix)>=0):
            alphav = cvx.Variable(num_users)
            obj = cvx.Minimize(cvx.quad_form(alphav, p_matrix))
            prob = cvx.Problem(obj, [cvx.sum(alphav) == 1.0, alphav >= 0])
            prob.solve()
            alpha = alphav.value
            alpha = [(i)*(i>eps) for i in alpha] # zero-out small weights (<eps)
        else:
            alpha = None # if no solution for the optimization problem, use local classifier only
        
        avg_weight.append(alpha)

    return avg_weight

# https://github.com/JianXu95/FedPAC/blob/main/tools.py#L10
def pairwise(data):
    n = len(data)
    for i in range(n):
        for j in range(i, n):
            yield (data[i], data[j])