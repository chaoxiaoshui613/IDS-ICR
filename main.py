from __future__ import division, print_function, absolute_import

import os
import sys
import datetime
import argparse
import pandas as pd
import faiss
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# 从 centroid 模块导入所有内容
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.optimize import linear_sum_assignment
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
import os
from collections import Counter
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
# torch.set_default_device('cpu')  # GPU version
# ============ GPU Setup ============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # domains
    parser.add_argument("--source", help="Source CSV file", default='source.csv')
    parser.add_argument("--target", help="Target CSV file", default='target.csv')
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--shared_classes", type=int, default=7, help="Number of classes of source domain -- known classes")
    parser.add_argument("--all_classes", type=int, default=8, help="Known+unknown classes")
    parser.add_argument("--log_dir", default='D:/paper_code/idea/dataset_da/log/', help="Path of the log folder")
    parser.add_argument("--data_dir", default='D:/paper_code/idea/dataset_da/cic/', help="Path of the dataset")
    parser.add_argument("--gpu", type=int, default=0, help="gpu chosen for the training")
    parser.add_argument("--use_VGG", action='store_true', default=False, help="If use VGG")
    parser.add_argument("--name", type=str, default='1')
    # 添加对 --f 参数的处理
    parser.add_argument("--f", type=str, help="Jupyter kernel connection file", default=None)

    return parser.parse_args()


args = get_args()

# 读取 CSV 数据
try:
    source_data = pd.read_csv(os.path.join(args.data_dir, args.source))
    target_data = pd.read_csv(os.path.join(args.data_dir, args.target))
    print("成功读取源数据和目标数据。")
except FileNotFoundError:
    print("未找到指定的 CSV 文件，请检查文件路径。")
    sys.exit(1)

orig_stdout = sys.stdout
max_iter = 10000
warmiter = 3

args.log_dir = args.log_dir + args.source.split('/')[-1].split('.')[0][0] + '2' + args.target.split('/')[-1].split('.')[0][0] + '_' + args.name

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# ============ Logging Setup ============
# 在脚本所在文件夹下创建 logs/ 目录保存日志
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, 'logs')
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

log_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_path = os.path.join(LOGS_DIR, f'training_{log_timestamp}.log')

class Tee:
    """同时写入控制台和日志文件。"""
    def __init__(self, file_path, mode='w'):
        self.file = open(file_path, mode, encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()

sys.stdout = Tee(log_file_path)

print('\n')
print('TRAIN START!')
print(f'Log file: {log_file_path}')
print('\n')
print('THE OUTPUT IS SAVED IN A TXT FILE HERE -------------------------------------------> ', args.log_dir)
print('\n')
    

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split

def one_hot(n_class, index):
    tmp = np.zeros((n_class,), dtype=np.float32)
    # print(f"Index value: {index}, type: {type(index)}")
    if isinstance(index, torch.Tensor):
        index = int(index.item())
    tmp[index] = 1
    return tmp
# 源数据集训练数据转换函数
def source_train_transform(data, label, is_train):

    label = one_hot(args.all_classes, label)
    return data, label


# 目标数据集训练数据转换函数
# 已知类标签(0 ~ shared_classes-1)映射为 one_hot(all_classes, label)
# 未知类标签映射为 one_hot(all_classes, shared_classes)
def target_train_transform(data, label, is_train):
    if label in range(args.shared_classes):
        label = one_hot(args.all_classes, label)
    else:
        label = one_hot(args.all_classes, args.shared_classes)
    return data, label


# 目标数据集测试数据转换函数
# 测试集标签范围可能超出 shared_classes（含未知攻击），
# 使用 max_test_label 作为 one_hot 维度确保不越界
_MAX_TEST_LABEL = 13  # CICIDS2017 测试集标签最大值为 11，取 13 留有安全裕量
def target_test_transform(data, label, is_train):
    label = one_hot(_MAX_TEST_LABEL, label)
    return data, label


# 获取数据集信息
def get_split_dataset_info(csv_file):
    try:
        df = pd.read_csv(csv_file)
        # 假设最后一列为标签列
        labels = df.iloc[:, -1].values
        data = df.iloc[:, :-1].values
        # 在初始化时就将数据转换为 torch.Tensor 类型
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        return data, labels
    except FileNotFoundError:
        print(f"文件 {csv_file} 未找到。")
        return [], []


# # 自定义数据集类
# class CustomDataset(Dataset):
#     def __init__(self, data, labels, data_transformer=None):
#         self.data = data
#         self.labels = labels
#         self.data_transformer = data_transformer

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         data_point = self.data[index]
#         label = self.labels[index]
#         if self.data_transformer:
#             data_point, label = self.data_transformer(data_point, label, is_train=True)
#         return data_point, label
class CustomDataset(Dataset):
    def __init__(self, data, labels, data_transformer=None):
        # 验证输入的数据和标签长度是否一致
        assert len(data) == len(labels), "The length of data and labels must be the same."
        self.data = data
        self.labels = labels
        self.data_transformer = data_transformer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index, is_train=True):
        # 检查索引是否越界
        assert 0 <= index < len(self.data), f"Index {index} is out of bounds for data of length {len(self.data)}."
        data_point = self.data[index]
        label = self.labels[index]
        if self.data_transformer:
            data_point, label = self.data_transformer(data_point, label, is_train)
        return data_point, label
# 源数据集训练数据加载
# source_csv_file = 'D:/paper_code/idea/dataset_da/cic/source_newcic_0_4.csv' 
source_csv_file = 'D:/data/CIC-IDS2017/train_0_6.csv'  # 请替换为实际的源数据集 CSV 文件路径
source_images, source_labels = get_split_dataset_info(source_csv_file)
print("Unique values in source:", np.unique(source_labels.flatten()))
ds = CustomDataset(source_images, source_labels, data_transformer=source_train_transform)
# 调整 DataLoader 参数
source_train = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

# 目标数据集训练数据加载
# target_train_csv_file = 'D:/paper_code/idea/dataset_da/cic/target_newcic_0_6.csv'  
target_train_csv_file = 'D:/data/CIC-IDS2017/test_0_8.csv'# 请替换为实际的目标数据集训练 CSV 文件路径
target_train_images, target_train_labels = get_split_dataset_info(target_train_csv_file)
# train_images, test_images, train_labels, test_labels = train_test_split(target_train_images, target_train_labels, test_size=0.2, random_state=42)
# print("Unique values in target:", np.unique(train_labels.flatten()))
ds1 = CustomDataset(target_train_images, target_train_labels, data_transformer=target_train_transform)
target_train = DataLoader(ds1, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
target_train_samples_count = len(target_train.dataset)
print(f"target_train 里样本的数量为: {target_train_samples_count}")
# 目标数据集测试数据加载
# target_test_csv_file = 'D:/paper_code/idea/dataset_da/cic/target_newcic_0_6.csv'
target_test_csv_file = 'D:/data/CIC-IDS2017/test_0_8.csv'  # 请替换为实际的目标数据集测试 CSV 文件路径
target_test_images, target_test_labels = get_split_dataset_info(target_test_csv_file)
ds2 = CustomDataset(target_test_images, target_test_labels, data_transformer=target_test_transform)
print(np.unique(target_train_labels.flatten()))
target_test = DataLoader(ds2, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)


import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from typing import Union

class Centroids(object):
    def __init__(self, class_num, dim, use_cuda):
        self.class_num = class_num
        self.src_ctrs = torch.ones((class_num, dim))
        self.tgt_ctrs = torch.ones((class_num, dim+1))
        self.unk_crts = torch.ones((class_num, 256))
        self.src_ctrs *= 1e-10
        self.tgt_ctrs *= 1e-10
        self.unk_crts *= 1e-10
        self.dim = dim
        if use_cuda:
            self.src_ctrs = self.src_ctrs.cuda()
            self.tgt_ctrs = self.tgt_ctrs.cuda()
            self.unk_crts = self.unk_crts.cuda()
            

    def get_centroids(self, domain=None, cid=None):
        if domain == 'source':
            return self.src_ctrs if cid is None else self.src_ctrs[cid, :]
        elif domain == 'target':
            return self.tgt_ctrs if cid is None else self.tgt_ctrs[cid, :]
        else:
            return self.src_ctrs, self.tgt_ctrs
    
    def get_virtual_centroids(self):
        return self.unk_crts
    

    @torch.no_grad()
    def update(self, pred_s, pred_t, label_s,label_unk=None, ):
        self.upd_src_centroids(pred_s, label_s)
        self.upd_tgt_centroids(pred_t, label_unk)

    @torch.no_grad()
    def update_virtual(self, feature_unk, label_unk):
        c_weight = torch.zeros(self.class_num)
        for i in range(self.class_num):
            if feature_unk[label_unk==i].shape[0]>=1:
                
                new_centroid = torch.mean(torch.tensor(feature_unk[label_unk==i]), 0).squeeze()
            # print(feature_unk[label_unk==i].shape)
                self.unk_crts[i, :] = new_centroid
                c_weight[i]+=feature_unk[label_unk==i].shape[0]
        
        c_weight = c_weight/torch.sum(c_weight)
        return c_weight
           
    @torch.no_grad()
    def upd_src_centroids(self, probs, labels):
        # feats = to_np(feats)
        #labels = to_np(labels)
        # last_centroids = to_np(self.src_ctrs)
        #probs = to_np(probs)
        
        for i in range(self.class_num):
            
            data_idx = np.argwhere(labels[:,i] == 1)[:,0]

            new_centroid = torch.mean(torch.tensor(probs[data_idx, :self.dim]), 0).squeeze()
            
            #from IPython import embed;embed()
            self.src_ctrs[i, :] = new_centroid
        

    @torch.no_grad()
    def upd_tgt_centroids(self, probs, labels):
        # feats = to_np(feats)
        # last_centroids = to_np(self.tgt_ctrs)
        # src_centroids = to_np(self.src_ctrs)
        #from IPython import embed;embed()
      
        if labels is None:
            return
        #pseudo_label = to_np(pseudo_label)
        #probs = to_np(probs)

        for i in range(self.class_num):
            
            data_idx = np.argwhere(labels==i)
            new_centroid = torch.mean(torch.tensor(probs[data_idx]), 0).squeeze()
            # if last_centroids[i] != np.zeros_like((1, feats.shape[0])):
            # print(cs)
            self.tgt_ctrs[i, :] = new_centroid


def crit_intra(feats, y, centers, lambd=1e-3):
    class_num = len(centers)
    batch_size = y.shape[0]

    expanded_centers = centers.expand(batch_size, -1, -1)
    expanded_feats = feats.expand(class_num, -1, -1).transpose(1, 0)
    # distance_centers = (expanded_feats - expanded_centers).pow(2).sum(dim=-1)
    distance_centers = cal_sim(expanded_feats, expanded_centers)
    distance_centers = distance_centers.reshape(batch_size, class_num)

    intra_distances = distance_centers.gather(1, y.unsqueeze(1))
    # intra_distances = distances_same.sum()
    inter_distances = distance_centers.sum(dim=-1) - intra_distances

    epsilon = 1e-6
    loss = (1 / 2.0 / batch_size / class_num) * intra_distances / \
           (inter_distances + epsilon)
    loss = loss.sum()
    loss *= lambd
    return loss


def crit_inter(center1, center2, lambd=1e-3):
    # dists = F.pairwise_distance(center1, center2)
    # loss = torch.mean(dists)

    # dists = cal_cossim(center1.cpu().numpy(), center2.cpu().numpy())
    dists = cal_sim(center1, center2)
    loss = 0
    for i in range(center1.shape[0]):
        loss += dists[i]#[i]
    loss /= center1.shape[0]
    loss *= lambd
    return loss, dists


def crit_contrast(feats, probs, s_ctds, t_ctds, lambd=1e-3):
    batch_num = feats.shape[0]
    class_num = s_ctds.shape[0]
    probs = F.softmax(probs, dim=-1)
    max_probs, preds = probs.max(1, keepdim=True)
    # print(probs.shape, max_probs.shape)
    select_index = torch.nonzero(max_probs.squeeze() >= 0.3).squeeze(1)
    select_index = select_index.cpu().tolist()

    # todo: calculate margins
    # dist_ctds = cal_cossim(to_np(s_ctds), to_np(t_ctds))
    dist_ctds = cal_sim(s_ctds, t_ctds)
    # print('dist_ctds', dist_ctds.shape)

    M = np.ones(class_num)
    for i in range(class_num):
        # M[i] = np.sum(dist_ctds[i, :]) - dist_ctds[i, i]
        M[i] = dist_ctds.mean() - dist_ctds[i]
        M[i] /= class_num - 1
    # print('M', M)

    # todo: calculate D_k between known samples to its source centroid &
    # todo: calculate D_u distances between unknown samples to all source centroids
    D_k, n_k = 0, 1e-5
    D_u, n_u = 0, 1e-5
    for i in select_index:
        class_id = preds[i][0]
        if class_id < class_num:
            # D_k += F.pairwise_distance(feats[i, :], s_ctds[class_id]).squeeze()
            # print(feats.shape, i)
            D_k += cal_sim(feats[i, :], s_ctds[class_id, :])
            # print('D_k', D_k)
            n_k += 1
        else:
            # todo: judge if unknown sample in the radius region of known centroid
            rp_feats = feats[i, :].unsqueeze(0).repeat(class_num, 1)

            # dist_known = F.pairwise_distance(rp_feats, s_ctds)
            dist_known = cal_sim(rp_feats, s_ctds)
            # print('dist_known', len(dist_known), dist_known)

            M_mean = M.mean()
            outliers = dist_known < M_mean
            dist_margin = (dist_known - M_mean) * outliers.float()
            D_u += dist_margin.sum()

    loss = D_k / n_k  # - D_u / n_u
    return loss.mean() * lambd


class BaseAutoencoder(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseAutoencoder, self).__init__()

    def output_num(self):
        pass


class TabularAutoencoder(BaseAutoencoder):
    def __init__(self, input_dim, hidden_dims=[78, 256], normalize=True):
        super(TabularAutoencoder, self).__init__()
        self.normalize = normalize
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # 编码器
        encoder_layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器
        decoder_layers = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            decoder_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(hidden_dims[-1], input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

        self.__in_features = hidden_dims[-1]

    def get_mean(self):
        if not hasattr(self, 'mean'):
            self.mean = torch.zeros(self.input_dim, dtype=torch.float32)
        return self.mean

    def get_std(self):
        if not hasattr(self, 'std'):
            self.std = torch.ones(self.input_dim, dtype=torch.float32)
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def output_num(self):
        return self.__in_features

class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256, temp=0.05):
        super(CLS, self).__init__()
        self.temp = 1
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.weight1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.fc = nn.Linear(bottle_neck_dim, out_dim, bias=False)

            self.main = nn.Sequential(
                self.bottleneck,
                nn.Sequential(
                    nn.BatchNorm1d(bottle_neck_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    self.fc
                ),
                nn.Softmax(dim=-1)
            )
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        out = [x]
        for i, module in enumerate(self.main.children()):
            if i == 0:
                x = module(x)
                x = x / torch.norm(x, dim=-1, keepdim=True)
            else:
                x = module(x)
            out.append(x)

        out[-2] = out[-2] / self.temp
        out[-1] = nn.Softmax(dim=-1)(out[-2])
        # print(len(out))
        return out

    def virt_forward(self, K, feature_source, logits: torch.Tensor, target: Union[torch.Tensor, None] = None):
        if self.training:
            if isinstance(K, np.ndarray):
                K = torch.from_numpy(K).to(torch.float32)

            # 确保 W_yi 也是 PyTorch 张量
            with torch.no_grad():
                W_yi = torch.gather(self.fc.weight, 0, target.unsqueeze(1).expand(target.size(0), self.fc.weight.size(1)))
                if isinstance(W_yi, np.ndarray):
                    W_yi = torch.from_numpy(W_yi).to(torch.float32)
                W_virt = torch.norm(W_yi, dim=1).unsqueeze(-1).unsqueeze(-1) * (
                        (K / torch.norm(K, dim=1).unsqueeze(-1)).unsqueeze(0))
            vir = torch.bmm(W_virt, feature_source.unsqueeze(-1)).squeeze(-1)
            logits = torch.cat([logits, vir], dim=-1)
            x = nn.Softmax(-1)(logits)
        return x

    
class AdversarialNetwork(nn.Module):
    def __init__(self):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential()
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x = self.grl(x)
        for module in self.main.children():
            x = module(x)
        return x
class LargeAdversarialNetwork(AdversarialNetwork):
    def __init__(self, in_feature):
        super(LargeAdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
        self.main = nn.Sequential(
            self.ad_layer1,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer2,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer3,
            self.sigmoid
        )

class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        return input
    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs
class GradientReverseModule(nn.Module):
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply
    def forward(self, x):
        self.coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        return self.grl(self.coeff, x)
# 定义一个简单的全连接神经网络模型
class AnomalyDetector(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


all_centroids = Centroids(class_num=args.shared_classes, dim=args.shared_classes, use_cuda=(DEVICE.type=="cuda"))
discriminator = LargeAdversarialNetwork(256)
feature_extractor = TabularAutoencoder(78)
cls = CLS(feature_extractor.output_num(), args.all_classes, bottle_neck_dim=256)
net = nn.Sequential(feature_extractor, cls)  
discriminator = discriminator.to(DEVICE)
feature_extractor = feature_extractor.to(DEVICE)
cls = cls.to(DEVICE)
net = net.to(DEVICE)


import numpy as np
import tensorflow as tf
import tensorlayer as tl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
import os
from collections import Counter
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd  # 新增，用于读取表格数据


class Accumulator(dict):
    def __init__(self, name_or_names, accumulate_fn=np.concatenate):
        super(Accumulator, self).__init__()
        self.names = [name_or_names] if isinstance(name_or_names, str) else name_or_names
        self.accumulate_fn = accumulate_fn
        for name in self.names:
            self.__setitem__(name, [])

    def updateData(self, scope):
        for name in self.names:
            if scope[name].shape[-1] > 0:
                self.__getitem__(name).append(scope[name])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb:
            print(exc_tb)
            return False

        for name in self.names:
            if len(self.__getitem__(name)) > 0:
                self.__setitem__(name, self.accumulate_fn(self.__getitem__(name)))

        return True
# class Accumulator(dict):
#     def __init__(self, name_or_names, accumulate_fn=np.concatenate):
#         super(Accumulator, self).__init__()
#         self.names = [name_or_names] if isinstance(name_or_names, str) else name_or_names
#         self.accumulate_fn = accumulate_fn
#         for name in self.names:
#             self.__setitem__(name, [])

#     def updateData(self, scope):
#         for name in self.names:
#             print(f"Checking data for {name}: shape={scope[name].shape}, last dim size={scope[name].shape[-1]}")
#             if scope[name].shape[-1] > 0:
#                 print(f"Adding data for {name}")
#                 self.__getitem__(name).append(scope[name])

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if exc_tb:
#             print(exc_tb)
#             return False

#         for name in self.names:
#             if len(self.__getitem__(name)) > 0:
#                 print(f"Before accumulation for {name}: {[arr.shape for arr in self.__getitem__(name)]}")
#                 self.__setitem__(name, self.accumulate_fn(self.__getitem__(name)))
#                 print(f"After accumulation for {name}: {self.__getitem__(name).shape}")

#         return True

class TrainingModeManager:
    def __init__(self, nets, train=False):
        self.nets = nets
        self.modes = [net.training for net in nets]
        self.train = train

    def __enter__(self):
        for net in self.nets:
            net.train(self.train)

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for (mode, net) in zip(self.modes, self.nets):
            net.train(mode)
        self.nets = None  # release reference, to avoid imexplicit reference
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True


def clear_output():
    def clear():
        return

    try:
        from IPython.display import clear_output as clear
    except ImportError as e:
        pass
    import os

    def cls():
        os.system('cls' if os.name == 'nt' else 'clear')

    clear()
    cls()


def addkey(diction, key, global_vars):
    diction[key] = global_vars[key]


def track_scalars(logger, names, global_vars):
    values = {}
    for name in names:
        addkey(values, name, global_vars)
    for k in values:
        values[k] = variable_to_numpy(values[k])
    for k, v in list(values.items()):
        logger.log_scalar(k, v)
    print(values)


def variable_to_numpy(x):
    ans = x.cpu().data.numpy()
    # if torch.numel(x) == 1:
    #     return float(np.sum(ans))
    return ans


def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))


def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(ans)





class OptimWithSheduler:
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr=g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1


class OptimizerManager:
    def __init__(self, optims):
        self.optims = optims  # if isinstance(optims, Iterable) else [optims]

    def __enter__(self):
        for op in self.optims:
            op.zero_grad()

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True


def setGPU(i):
    global os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % (i)
    gpus = [x.strip() for x in (str(i)).split(',')]
    NGPU = len(gpus)
    print(('gpu(s) to be used: %s' % str(gpus)))
    return NGPU


class Logger(object):
    def __init__(self, log_dir, clear=False):
        if clear:
            os.system('rm %s -r' % log_dir)
        tl.files.exists_or_mkdir(log_dir)
        self.writer = tf.summary.create_file_writer(log_dir)
        self.step = 0
        self.log_dir = log_dir

    def log_scalar(self, tag, value, step=None):
        if not step:
            step = self.step
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag,
                                                                         simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_images(self, tag, images, step=None):
        if not step:
            step = self.step

        im_summaries = []
        for nr, img in enumerate(images):
            s = StringIO()

            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)

            if img.shape[-1] == 1:
                img = np.tile(img, [1, 1, 3])
            img = to_rgb_np(img)
            plt.imsave(s, img, format='png')

            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_histogram(self, tag, values, step=None, bins=1000):
        if not step:
            step = self.step
        values = np.array(values)
        counts, bin_edges = np.histogram(values, bins=bins)
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_bar(self, tag, values, xs=None, step=None):
        if not step:
            step = self.step

        values = np.asarray(values).flatten()
        if not xs:
            axises = list(range(len(values)))
        else:
            axises = xs
        hist = tf.HistogramProto()
        hist.min = float(min(axises))
        hist.max = float(max(axises))
        hist.num = sum(values)
        hist.sum = sum([y * x for (x, y) in zip(axises, values)])
        hist.sum_squares = sum([y * (x ** 2) for (x, y) in zip(axises, values)])

        for edge in axises:
            hist.bucket_limit.append(edge - 1e-10)
            hist.bucket_limit.append(edge + 1e-10)
        for c in values:
            hist.bucket.append(0)
            hist.bucket.append(c)

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, self.step)
        self.writer.flush()


class LossCounter:
    def __init__(self):
        self.ce = 0.0
        self.entropy = 0.0
        self.virtual = 0.0
        self.ce_ep = 0.0
        self.adv = 0.0
        self.batch = 0

    def addOntBatch(self, ce, entropy, virtual, ce_ep, adv):
        self.batch += 1
        self.ce += ce.item()
        self.entropy += entropy.item()
        self.virtual += virtual.item()
        self.ce_ep += ce_ep.item()
        self.adv += adv.item()


class AccuracyCounter:
    def __init__(self):
        self.Ncorrect = 0.0
        self.Ntotal = 0.0

    def addOntBatch(self, predict, label):
        assert predict.shape == label.shape
        correct_prediction = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
        Ncorrect = np.sum(correct_prediction.astype(np.float32))
        Ntotal = len(label)
        self.Ncorrect += Ncorrect
        self.Ntotal += Ntotal
        return Ncorrect / Ntotal

    def reportAccuracy(self):
        return np.asarray(self.Ncorrect, dtype=float) / np.asarray(self.Ntotal, dtype=float)


def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    if label.shape != predict_prob.shape:
        # this means that the target data shape is (B,)
        label = torch.zeros_like(predict_prob).scatter(1, label.unsqueeze(1), 1)

    N, C = label.size()
    N_, C_ = predict_prob.size()

    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)


def CrossEntropyLabelSmooth(targets, inputs, class_level_weight=None, instance_level_weight=None, epsilon=0.1):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    if targets.shape != inputs.shape:
        # this means that the target data shape is (B,)
        targets = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
    N, C = targets.size()
    log_probs = torch.log(inputs + 1e-12)

    if inputs.shape != targets.shape:
        # this means that the target data shape is (B,)
        targets = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    targets = (1 - epsilon) * targets + epsilon / C
    ce = (- targets * log_probs)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)


def BCELossForMultiClassification(label, predict_prob, class_level_weight=None, instance_level_weight=None,
                                  epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()

    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    bce = -label * torch.log(predict_prob + epsilon) - (1.0 - label) * torch.log(1.0 - predict_prob + epsilon)
    return torch.sum(instance_level_weight * bce * class_level_weight) / float(N)


def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):
    N, C = predict_prob.size()

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)


def plot_confusion_matrix(cm, true_classes, pred_classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
    pred_classes = pred_classes or true_classes
    if normalize:
        cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    true_tick_marks = np.arange(len(true_classes))
    plt.yticks(true_classes, true_classes)
    pred_tick_marks = np.arange(len(pred_classes))
    plt.xticks(pred_tick_marks, pred_classes)


# def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
#     if not true_labels:
#         true_labels = sorted(list(set(list(y_true))))
#     true_label_to_id = {x: i for (i, x) in enumerate(true_labels)}
#     if not pred_labels:
#         pred_labels = true_labels
#     pred_label_to_id = {x: i for (i, x) in enumerate(pred_labels)}
#     confusion_matrix = np.zeros([len(true_labels), len(pred_labels)])
#     for (true, pred) in zip(y_true, y_pred):
#         confusion_matrix[true_label_to_id[true]][pred_label_to_id[pred]] += 1.0
#     return confusion_matrix
def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
    if not true_labels:
        true_labels = sorted(list(set(list(y_true))))
    true_label_to_id = {x: i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x: i for (i, x) in enumerate(pred_labels)}
    confusion_matrix = np.zeros([len(true_labels), len(pred_labels)], dtype=int)
    for (true, pred) in zip(y_true, y_pred):
        confusion_matrix[true_label_to_id[true]][pred_label_to_id[pred]] += 1
    return confusion_matrix


def to_np(x):
    return x.squeeze().cpu().detach().numpy()


from sklearn.metrics.pairwise import cosine_similarity


def get_features(data_loader, model):
    model.eval()
    feats, labels = [], []
    probs, preds = [], []
    for batch_idx, batch_data in enumerate(data_loader):
        input, label = batch_data
        # input, label = input.cuda(), label.cuda(non_blocking=True)  # 移除这行
        feat, prob = model(input)
        prob, pred = prob.max(1, keepdim=True)

        feats.append(feat.cpu().detach().numpy())
        labels.append(label.cpu().detach().numpy())
        probs.append(prob.cpu().detach().numpy())
        preds.append(pred.cpu().detach().numpy())

    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    probs = np.concatenate(probs, axis=0)
    preds = np.concatenate(preds, axis=0)
    return feats, labels, probs, preds


def get_src_centroids(data_loader, model, args):
    feats, labels, probs, preds = get_features(data_loader, model)
    centroids = []
    for i in range(args.class_num - 1):
        data_idx = np.unique(np.argwhere(labels == i))
        feats_i = feats[data_idx].squeeze()

        center_i = np.mean(feats_i, axis=0)
        centroids.append(center_i)

    centroids = np.array(centroids).squeeze()
    return torch.from_numpy(centroids)  # 移除 .cuda()


def get_tgt_centroids(data_loader, model, th, src_centroids, args):
    feats, labels, probs, preds = get_features(data_loader, model)
    src_centroids = to_np(src_centroids)
    tgt_dissim = cal_sim(src_centroids, feats, rev=True)
    centroids = []
    for i in range(args.CLASS_NUM - 1):
        class_idx = np.unique(np.argwhere(preds == i))
        easy_idx = np.unique(np.argwhere(tgt_dissim[i, :] <= th))
        data_idx = np.intersect1d(class_idx, easy_idx)
        if len(data_idx) > 1:
            feats_i = feats[data_idx].squeeze()
        else:
            feats_i = np.zeros_like(feats)
            print(i, 'none')
        center_i = np.mean(feats_i, axis=0)
        centroids.append(center_i)

    centroids = np.array(centroids).squeeze()
    return torch.from_numpy(centroids)  # 移除 .cuda()


def upd_src_centroids(feats, labels, probs, last_centroids, args):
    new_centroids = []
    feats = to_np(feats)
    labels = to_np(labels)
    last_centroids = to_np(last_centroids)
    probs = F.softmax(probs, dim=1)
    probs = to_np(probs)
    for i in range(args.class_num - 1):
        if np.sum(labels == i) > 0:
            data_idx = np.intersect1d(np.argwhere(labels == i), np.argwhere(probs[:, i] > 0.1))
            new_centroid = np.mean(feats[data_idx], axis=0).reshape(1, -1)
            cs = cosine_similarity(new_centroid, last_centroids[i].reshape(1, -1))[0][0]
            new_centroid = cs * new_centroid + (1 - cs) * last_centroids[i]
        else:
            new_centroid = last_centroids[i]

        new_centroids.append(new_centroid.squeeze())

    new_centroids = np.array(new_centroids)
    return torch.from_numpy(new_centroids)  # 移除 .cuda()


def upd_tgt_centroids(feats, probs, last_centroids, src_centroids, args):
    new_centroids = []
    feats = to_np(feats)
    last_centroids = to_np(last_centroids)
    src_centroids = to_np(src_centroids)
    _, ps_labels = probs.max(1, keepdim=True)
    ps_labels = to_np(ps_labels)
    probs = F.softmax(probs, dim=1)
    probs = to_np(probs)
    for i in range(args.CLASS_NUM - 1):
        if np.sum(ps_labels == i) > 0:
            data_idx = np.intersect1d(np.argwhere(ps_labels == i), np.argwhere(probs[:, i] > 0.1))
            new_centroid = np.mean(feats[data_idx], axis=0).reshape(1, -1)

            if last_centroids[i] != np.zeros_like((1, feats.shape[0])):
                cs = cosine_similarity(new_centroid, src_centroids[i].reshape(1, -1))[0][0]
                new_centroid = cs * new_centroid + (1 - cs) * last_centroids[i]
        else:
            new_centroid = last_centroids[i]

        new_centroids.append(new_centroid.squeeze())

    new_centroids = np.array(new_centroids)
    return torch.from_numpy(new_centroids)  # 移除 .cuda()


def cal_sim(x1, x2, metric='cosine'):
    # x = x1.clone()
    if len(x1.shape) != 2:
        x1 = x1.reshape(-1, x1.shape[-1])
    if len(x2.shape) != 2:
        x2 = x2.reshape(-1, x2.shape[-1])

    if metric == 'cosine':
        sim = (F.cosine_similarity(x1, x2) + 1) / 2
    else:
        sim = F.pairwise_distance(x1, x2) / torch.norm(x2, dim=1)
    return sim
from sklearn.metrics import roc_curve, auc
def calculate_open_set_metrics(cm, shared_classes):
    # 确保混淆矩阵已归一化
    cm_normalized = cm.astype(float) / np.sum(cm, axis=1, keepdims=True)
    
    # 1. 计算OSCR (Open Set Classification Rate)
    # 已知类的平均分类准确率
    acc_os_star = sum([cm_normalized[i][i] for i in range(shared_classes)]) / shared_classes
    
    # 未知类的混淆矩阵部分（最后一行）
    unknown_class_row = cm_normalized[-1, :]
    
    # 正确预测为未知类的样本比例（最后一行最后一列）
    correct_unknown = unknown_class_row[-1]
    
    # OSCR计算: 已知类准确率和未知类拒绝率的平均值
    oscr = (acc_os_star + correct_unknown) / 2
    
    # 2. 准备计算AUROC和FPR@95所需的置信度和标签
    # 已知类样本的最大预测概率（对角线元素）
    known_confidences = np.diag(cm_normalized[:shared_classes, :shared_classes])
    
    # 未知类样本的预测概率分布（排除最后一列，即排除预测为未知类的概率）
    known_class_predictions = unknown_class_row[:-1]
    
    # 检查已知类预测分布是否有效（即是否有非零概率）
    n_unknown_samples = 1000 
    if np.sum(known_class_predictions) <= 0:
        # 如果所有未知类样本都被正确分类为未知类，手动设置置信度
        unknown_confidences_expanded = np.zeros(n_unknown_samples)
    else:
        # 归一化已知类预测分布
        normalized_p = known_class_predictions / np.sum(known_class_predictions)
        
        # 采样已知类索引（范围从0到shared_classes-1）
        # 用于计算的样本数量
        sampled_indices = np.random.choice(
            np.arange(shared_classes),  # 修正：范围从0到shared_classes-1
            size=n_unknown_samples,
            p=normalized_p
        )
        
        # 获取对应的置信度
        unknown_confidences_expanded = np.array([
            cm_normalized[-1, i] for i in sampled_indices
        ])
    
    # 创建标签: 1表示已知类，0表示未知类
    labels = np.hstack([
        np.ones_like(known_confidences),
        np.zeros_like(unknown_confidences_expanded)
    ])
    confidences = np.hstack([
        known_confidences,
        unknown_confidences_expanded
    ])
    
    # 3. 计算AUROC
    fpr, tpr, _ = roc_curve(labels, confidences)
    auroc = auc(fpr, tpr)
    
    # 4. 计算FPR@95TPR
    tpr_95_idx = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95tpr = fpr[tpr_95_idx]
    print(auroc)
    return auroc,fpr_at_95tpr, oscr
    


import torch
import numpy as np
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import faiss  # 导入 faiss 库
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from scipy.stats import entropy
# 定义一个简单的全连接神经网络模型
class AnomalyDetector(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyDetector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # 编码器部分，增加了层数和每层的神经元数量
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # 解码器部分，增加了层数和每层的神经元数量
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DomainBus(object):
    def __init__(self, domainloaders, train_samplers=None, iter_num=-1):
        self.domainloaders = domainloaders
        self.train_samplers = train_samplers
        self.domainiters = [iter(dataloader) for dataloader in self.domainloaders]
        self.domain_sizes = [len(dataloader) for dataloader in self.domainloaders]

        # 以目标域数据加载器的长度作为最大迭代次数
        target_loader_index = 1  # 假设 target_train 是 domainloaders 中的第二个元素
        self.max_iter_num = len(self.domainloaders[target_loader_index])
        self.current_iter = 0

    def get_samples(self):
        batch_split = []
        for i in range(len(self.domainloaders)):
            try:
                imgs, trgs = next(self.domainiters[i])
            except StopIteration:
                self.domainiters[i] = iter(self.domainloaders[i])
                imgs, trgs = next(self.domainiters[i])
            batch_split.append((imgs, trgs))
        self.current_iter += 1
        return batch_split

    def __len__(self):
        return self.max_iter_num

    def reset(self):
        self.current_iter = 0

    def __next__(self):
        if self.current_iter >= self.max_iter_num:
            raise StopIteration
        return self.get_samples()

    def __iter__(self):
        return self

    def __str__(self):
        return "\n".join([domainloader.__str__() for domainloader in self.domainloaders])

    def set_epoch(self, epoch):
        if self.train_samplers:
            for sampler in self.train_samplers:
                sampler.set_epoch(epoch)
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x
target_train_samples_count = len(target_train.dataset)
print(f"target_train 里样本的数量为: {target_train_samples_count}")
customgenearator = DomainBus([source_train, target_train])
total_iterations = len(customgenearator)

# 计算每个数据加载器每次迭代返回的样本数量
total_samples_per_iteration = 0

# 计算 customgenearator 中的样本总数
total_samples = total_iterations * target_train.batch_size  # 只计算目标域的样本总数
total_label_target_count =0
print(f"customgenearator 中目标域的样本总数为: {total_samples}")
with torch.no_grad():
    with Accumulator(['fs', 'ft', 'ls', 'lt']) as ProbRecorder:
        for i, ((data_source, label_source), (data_target, label_target)) in enumerate(customgenearator):
            _, feature_source, fc_source, predict_prob_source = net(data_source)
            ft1, feature_target, fc_target, predict_prob_target = net(data_target)
            fs, ft, ls, lt = [variable_to_numpy(x) for x in (
                feature_source, feature_target, torch.nonzero(label_source, as_tuple=True)[1],
                torch.nonzero(label_target, as_tuple=True)[1])]
            ProbRecorder.updateData(globals())
            current_count = len(label_target)
            total_label_target_count += current_count

source_features = torch.tensor(ProbRecorder['fs'], dtype=torch.float32)
source_labels_orig = np.array(ProbRecorder['ls'])  # 保留原始多类标签用于逐类质心计算
source_labels_binary = (source_labels_orig != 0).astype(int)
source_labels = torch.tensor(source_labels_binary, dtype=torch.float32).unsqueeze(1)

batch_size = 64
classifier_dataset = TensorDataset(source_features, source_labels)
classifier_dataloader = DataLoader(classifier_dataset, batch_size=batch_size, shuffle=True)

# 初始化神经网络模型
input_dim = source_features.size(1)
model = BinaryClassifier(input_dim)
criterion = nn.BCELoss()  # 二元交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 训练神经网络模型
num_epochs = 30
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (features, labels) in enumerate(classifier_dataloader):
        outputs = model(features)
        # 在训练循环中打印输出的极值
        # print("模型输出的最小值：", outputs.min().item())
        # print("模型输出的最大值：", outputs.max().item())
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(classifier_dataloader):.4f}')

# 使用训练好的模型对目标域数据进行预测
target_features = torch.tensor(ProbRecorder['ft'], dtype=torch.float32)
print("target_feature", len(target_features))
with torch.no_grad():
    target_predictions = model(target_features)
    predicted_labels = (target_predictions > 0.5).float().numpy().flatten()

    # 处理目标域真实标签，将0设为0（正常），其他设为1（异常）
    true_labels = np.array(ProbRecorder['lt'])
    true_binary_labels = (true_labels != 0).astype(int)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(true_binary_labels, predicted_labels)
    print("Confusion Matrix:")
    print("           Predicted Negative  Predicted Positive")
    print(f"Actual Negative      {conf_matrix[0, 0]}               {conf_matrix[0, 1]}")
    print(f"Actual Positive      {conf_matrix[1, 0]}               {conf_matrix[1, 1]}")
    # 找出被识别为0的样本
    predicted_zero_indices = np.where(predicted_labels == 0)[0]
    predicted_zero_features = target_features[predicted_zero_indices]
# 找出被识别为非 0 的样本
predicted_non_zero_indices = np.where(predicted_labels != 0)[0]
predicted_non_zero_features = target_features[predicted_non_zero_indices]

# 对正常目标流量（预测为 0）聚成一类
normal_target_centroid = predicted_zero_features.mean(dim=0).cpu().numpy() if len(predicted_zero_features) > 0 else None

# 对异常目标流量（预测不为 0）进行 K_cluster - 1 聚类
K_cluster = 8
if len(predicted_non_zero_features) > 0:
    faiss_kmeans = faiss.Kmeans(input_dim, int(K_cluster - 1), niter=800, verbose=False, min_points_per_centroid=1, gpu=False)
    faiss_kmeans.train(predicted_non_zero_features.cpu().numpy())
    abnormal_t_centroids = faiss_kmeans.centroids
else:
    abnormal_t_centroids = np.array([])

# 合并正常和异常目标聚类质心
if normal_target_centroid is not None:
    t_centroids = np.vstack([normal_target_centroid.reshape(1, -1), abnormal_t_centroids])
else:
    t_centroids = abnormal_t_centroids

# 计算源域每个已知攻击类的质心（排除正常类0，仅用攻击类1..shared_classes-1）
s_attack_centroids = []
for i in range(1, args.shared_classes):  # 从类1开始，排除正常流量
    class_indices = (source_labels_orig == i)
    class_features = source_features[class_indices]
    if len(class_features) > 0:
        s_attack_centroids.append(class_features.mean(dim=0).cpu().numpy())
s_attack_centroids = np.stack(s_attack_centroids, axis=0) if s_attack_centroids else np.array([])

# 二分图匹配：已知攻击类质心 (C-1个) vs 目标域异常聚类质心 (K_cluster-1个)
# 源域攻击类数: args.shared_classes-1 = 6, 目标域异常簇数: K_cluster-1 = 7
# 必然存在未匹配的簇 → 即为未知攻击的初始虚拟类表示
if len(s_attack_centroids) > 0 and len(abnormal_t_centroids) > 0:
    cost = np.linalg.norm(s_attack_centroids[:, None, :] - abnormal_t_centroids[None, :, :], axis=-1)
    _, t_match = linear_sum_assignment(cost)
    nomatch = []
    for i in range(K_cluster - 1):
        if i not in t_match:
            nomatch.append(abnormal_t_centroids[i])
    nomatch = np.stack(nomatch, axis=0) if nomatch else np.array([])
else:
    nomatch = np.array([])

# 打印匹配信息
print(f"源域攻击类质心数: {len(s_attack_centroids)}, 目标域异常簇数: {len(abnormal_t_centroids)}")
print(f"匹配结果: {len(t_match)} 对匹配, 未匹配簇数: {len(nomatch)}")

print("未匹配的目标域聚类质心：", nomatch)
  
del (ProbRecorder)

scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=max_iter)
optimizer_discriminator = OptimWithSheduler(
    optim.SGD(discriminator.parameters(), lr=args.learning_rate * 10, weight_decay=5e-4, momentum=0.9, nesterov=True),
    scheduler)
optimizer_feature_extractor = OptimWithSheduler(
    optim.SGD(feature_extractor.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),
    scheduler)
optimizer_cls = OptimWithSheduler(
    optim.SGD(cls.parameters(), lr=args.learning_rate * 10, weight_decay=5e-4, momentum=0.9, nesterov=True),
    scheduler)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
best_model_params = {}
best_experiment_info = {}
epoch = 0
k = 0
best_os = 0
best_os_star = 0
best_unk = 0
best_hos = 0
best_epoch = 0
# c_weight = torch.ones(args.shared_classes)

while epoch < 100:
    customgenearator = DomainBus([source_train, target_train])
    losscounter = LossCounter()
    with Accumulator(['pred_s', 'pred_t', 'label_s', 'kl', 'fss', 'ftt']) as ProbRecorder:
        for i, ((im_source, label_source), (im_target, label_target)) in enumerate(customgenearator):
            # 打印源域和目标域训练标签的唯一值)

            _, feature_source, fc_source, predict_prob_source = net.forward(im_source)
            ft1, feature_target, fc_target, predict_prob_target = net.forward(im_target)

            domain_prob_discriminator_1_source = discriminator.forward(feature_source)
            domain_prob_discriminator_1_target = discriminator.forward(feature_target)

            s_ctds, t_ctds = all_centroids.get_centroids()
            _, pseudo_t_label = predict_prob_target[:, :args.shared_classes].max(1)
            # 打印伪标签的唯一值
            # print(f"Epoch {epoch}, Iter {i}: Pseudo target label unique values:", np.unique(pseudo_t_label))

            kltarget = torch.nn.functional.kl_div((nn.Softmax(-1)(fc_target[:, :args.shared_classes])).log(), s_ctds[pseudo_t_label], reduction='none').sum(1).detach()
            kltarget = torch.where(torch.isinf(kltarget), torch.full_like(kltarget, 10), kltarget)

            # 每batch重拟合GMM，使已知/未知判别随特征空间演化而更新
            gmm = GaussianMixture(n_components=2, covariance_type='full', n_init=1).fit(to_np(kltarget)[:, None])

            known_cluster = np.argmin(gmm.means_)
            unknown_cluster = np.argmax(gmm.means_)
            gmm_index = gmm.predict(to_np(kltarget)[:, None])

            pred_s, pred_t, label_s, kl, fss, ftt \
                = [variable_to_numpy(x) for x in (nn.Softmax(-1)(fc_source[:, :args.shared_classes]),
                                                   predict_prob_target, label_source, kltarget, feature_source, feature_target)]
            ProbRecorder.updateData(globals())

            weight = gmm.predict_proba(to_np(kltarget)[:, None])[:, known_cluster]
            weight = torch.tensor(weight)
            weight = weight.detach()

            if epoch <= 10:  # first 10 epoch use most confident sample
                weight = torch.where(weight > 0.8, torch.tensor([1]).float(), torch.tensor([0]).float()).detach()
                r = torch.nonzero(torch.tensor(gmm_index != known_cluster))
                topk = 16
                if r.size()[0] > topk:
                    r = torch.sort(kltarget.detach(), dim=0)[1][-1 * topk:]
            else:
                weight = torch.where(torch.tensor(gmm_index == known_cluster), torch.tensor([1]).float(), torch.tensor([0]).float()).detach()
                r = torch.nonzero(torch.tensor(gmm_index == unknown_cluster))

            feature_otherep = torch.index_select(ft1, 0, r.view(-1))
            if r.size()[0] > 1:
                _, feature_otherep, logits_otherep, predict_prob_otherep = cls.forward(feature_otherep)
                _, pseudo_index = predict_prob_otherep[:, args.shared_classes:].max(1)
                pseudo_index = pseudo_index + args.shared_classes

                pseudo_label = torch.zeros(r.size()[0], args.all_classes)
                pseudo_label = pseudo_label.scatter_(1, pseudo_index.unsqueeze(1), torch.ones(r.size()[0], 1))
                ce_ep = CrossEntropyLoss(pseudo_label[:, :], predict_prob_otherep[:, :])
            else:
                ce_ep = torch.tensor(0.0)
            if isinstance(nomatch, np.ndarray):
                nomatch = torch.from_numpy(nomatch)

            ce = CrossEntropyLoss(label_source, nn.Softmax(-1)(fc_source))

            # 当有未匹配的虚拟类质心时才计算 L_virt，否则跳过
            if nomatch is not None and nomatch.numel() > 0:
                virtual_predict_prob_source = cls.virt_forward(nomatch, feature_source, fc_source[:, :],
                                                            torch.nonzero(label_source)[:, 1])
                p = torch.zeros([label_source.shape[0], nomatch.size(0)])
                v_label_source = torch.cat((label_source[:, :], p), 1)
                virtual_ce = CrossEntropyLoss(v_label_source, virtual_predict_prob_source)
            else:
                virtual_ce = torch.tensor(0.0)
        

            entropy = EntropyLoss(predict_prob_target[:, :], instance_level_weight=weight.contiguous())

            adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source),
                                                     predict_prob=domain_prob_discriminator_1_source)
            adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target),
                                                      predict_prob=1 - domain_prob_discriminator_1_target,
                                                      instance_level_weight=weight.contiguous())

            with OptimizerManager([optimizer_cls, optimizer_feature_extractor, optimizer_discriminator]):
                if epoch <= warmiter:
                    loss = 1 * ce + 1 * virtual_ce + 0 * adv_loss + 0 * entropy + 0 * ce_ep
                else:
                    loss = ce + 0.3 * virtual_ce + 0.5* adv_loss + 0.5 * entropy + 0.5* ce_ep
                loss.backward()
            losscounter.addOntBatch(ce, entropy, virtual_ce, ce_ep, adv_loss)
            k += 1

    all_centroids.update(ProbRecorder['pred_s'], ProbRecorder['pred_t'], ProbRecorder['label_s'])

    s_centroids = []
    for i in range(args.shared_classes):
        s_centroids.append(ProbRecorder['fss'][np.nonzero(ProbRecorder['label_s'])[1] == i].mean(axis=0))
    s_centroids = np.stack(s_centroids, axis=0)

    faiss_kmeans = faiss.Kmeans(256, int(K_cluster), niter=800, verbose=False, min_points_per_centroid=1, gpu=False)
    faiss_kmeans.train(ProbRecorder['ftt'])
    t_centroids = faiss_kmeans.centroids

    # find nomatched target cluster
    cost = np.linalg.norm(s_centroids[:, None, :] - t_centroids[None, :, :], axis=-1)
    _, t_match = linear_sum_assignment(cost)
    nomatch = []
    for i in range(K_cluster):
        if i not in t_match:
            nomatch.append(t_centroids[i])
    nomatch = np.stack(nomatch, axis=0)
    nomatch = torch.from_numpy(nomatch).detach().clone()

    if epoch == warmiter:
        # cluster shared class+K
        faiss_kmeans = faiss.Kmeans(256, int(args.all_classes), niter=800, verbose=False, min_points_per_centroid=1, gpu=False)
        faiss_kmeans.train(ProbRecorder['ftt'])

        t_centroids = faiss_kmeans.centroids
        cost = np.linalg.norm(s_centroids[:, None, :] - t_centroids[None, :, :], axis=-1)
        _, t_match = linear_sum_assignment(cost)
        # no match as unk weight
        init_unk_weight = []
        for i in range(args.all_classes):
            if i not in t_match:
                init_unk_weight.append(t_centroids[i])
        init_unk_weight = np.stack(init_unk_weight, axis=0)

        for key, v in net.state_dict().items():
            if key == '1.main.1.2.weight':
                v.requires_grad = False
                net.state_dict()['1.fc.weight'].requires_grad = False

                vvnorm = (torch.norm(v, dim=-1)).mean().cpu().numpy()
                init_unk_weight = init_unk_weight / np.linalg.norm(init_unk_weight, axis=-1, keepdims=True) * vvnorm
                fcweight = np.concatenate([v[:args.shared_classes].clone().detach().cpu().numpy(), init_unk_weight, ], axis=0)
                param = torch.from_numpy(fcweight).detach().clone()
                net.state_dict()['1.fc.weight'].copy_(param)

                v.requires_grad = True
                net.state_dict()['1.fc.weight'].requires_grad = True

    # 每epoch用BayesianGMM拟合KL散度分布，完成已知/未知划分
    gmm = BayesianGaussianMixture(n_components=2, max_iter=800).fit(ProbRecorder['kl'][:, None])


    with TrainingModeManager([feature_extractor, cls], train=False) as mgr, Accumulator(['predict_prob', 'predict_index', 'label']) as accumulator:
        for (i, (im, label)) in enumerate(target_test):
            # 打印测试标签的唯一值
            # print(f"Epoch {epoch}, Test Iter {i}: Target test label unique values:", np.unique(label))

            ss, fs, _, predict_prob = net.forward(im)
            predict_prob, label = [variable_to_numpy(x) for x in (predict_prob, label)]
            label = np.argmax(label, axis=-1).reshape(-1, 1)
            # 打印经过 argmax 处理后的标签唯一值
            predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
            accumulator.updateData(globals())

    for x in list(accumulator.keys()):
        globals()[x] = accumulator[x]

    y_true = label.flatten()
    y_pred = predict_index.flatten()
    # print(f"Epoch {epoch}: Unique values in y_true:", np.unique(y_true))
    # print(f"Epoch {epoch}: Unique values in y_pred:", np.unique(y_pred))
    print("y_true dtype:", y_true.dtype)
    print("y_true unique values:", np.unique(y_true))
    print("y_pred dtype:", y_pred.dtype)
    print("y_pred unique values:", np.unique(y_pred))
    # 构建混淆矩阵：true_labels覆盖数据中所有可能的标签值
    # 已知类在 0..shared_classes-1，未知类标签可能超出此范围
    y_true_max = int(y_true.max())
    m = extended_confusion_matrix(y_true, y_pred,
        true_labels=list(range(y_true_max + 1)),
        pred_labels=list(range(args.all_classes)))
    print("整数形式的混淆矩阵 m:")
    # print(m)
    m_merged = np.copy(m)  # 创建副本避免修改原矩阵
    # 将所有未知类行（shared_classes 及之后的行）合并为一行
    num_unknown_rows = m_merged.shape[0] - args.shared_classes
    if num_unknown_rows > 1:
        # 合并所有未知行到 shared_classes 行
        m_merged[args.shared_classes, :] = m_merged[args.shared_classes:, :].sum(axis=0)
        # 删除多余的未知行，只保留 shared_classes + 1 行
        m_merged = m_merged[:args.shared_classes + 1, :]
    print(m_merged)
    # 计算概率混淆矩阵（按行归一化）
    m_prob = m_merged.astype(np.float64)  # 转换为浮点数避免整数除法
    row_sums = m_prob.sum(axis=1)[:, np.newaxis]  # 计算每行总和
    m_prob = np.divide(m_prob, row_sums, out=np.zeros_like(m_prob), where=(row_sums != 0))  # 避免除以零

    # 可视化概率混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(m_prob, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=list(range(m_prob.shape[1])),  # x轴标签为预测类别
                yticklabels=list(range(m_prob.shape[0])))  # y轴标签为真实类别
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Probabilistic Confusion Matrix')
    plt.show()

    # 打印概率混淆矩阵（保留两位小数）
    print("概率混淆矩阵:")
    print(np.round(m_prob, 3))
    cm = m_merged
    cm = cm.astype(float) / np.sum(cm, axis=1, keepdims=True)
    acc_os_star = sum([cm[i][i] for i in range(args.shared_classes)]) / args.shared_classes
    unknown_classes_matrix = cm[-1:]

    # 正确预测为未知类的样本数量（最后一列之和）
    correct_predictions = unknown_classes_matrix[:, -1].sum()

    # 未知类的总样本数量（后四行所有列之和）
    total_unknown_samples = unknown_classes_matrix.sum()

    # 计算未知类准确率
    unkn = correct_predictions / total_unknown_samples
    # 总体准确率按各类别等权平均：known * shared_classes + unknown * 1
    acc_os = (acc_os_star * args.shared_classes + unkn) / (args.shared_classes + 1)

    # HOS: 已知类和未知类准确率的调和平均 (Equation 19)
    if (acc_os_star + unkn) == 0:
        hos = 0.0
    else:
        hos = (2 * acc_os_star * unkn) / (acc_os_star + unkn)
    ce = losscounter.ce / losscounter.batch
    entropy = losscounter.entropy / losscounter.batch
    virtual = losscounter.virtual / losscounter.batch
    ce_ep = losscounter.ce_ep / losscounter.batch
    adv = losscounter.adv / losscounter.batch
    print('Epoch:{}\tOS: {:.3f}\tOS*:{:.3f}\tUnk:{:.3f}\tHos:{:.3f}\tce: {:.3f}\tentropy:{:.3f}\tvirtual:{:.3f}\tce_ep:{:.3f}\tadv:{:.3f}'.format(
        epoch, acc_os, acc_os_star, unkn, hos, ce, entropy, virtual, ce_ep, adv))
    def get_optimizer_state(optimizer_obj):
    
        if hasattr(optimizer_obj, 'optimizer'):
            return optimizer_obj.optimizer.state_dict()
        else:
            print(f"警告: 无法获取优化器状态，类型: {type(optimizer_obj)}")
            return None

    def load_optimizer_state(optimizer_obj, state_dict):
        """加载 OptimWithSheduler 包装的优化器状态"""
        if state_dict is None:
            return
        
        if hasattr(optimizer_obj, 'optimizer'):
            optimizer_obj.optimizer.load_state_dict(state_dict)
        else:
            print(f"警告: 无法加载优化器状态，类型: {type(optimizer_obj)}")

    if hos > best_hos:
        best_os = acc_os
        best_os_star = acc_os_star
        best_unk = unkn
        best_hos = hos
        best_epoch = epoch
        best_model_params['net_state_dict'] = net.state_dict()
        best_model_params['discriminator_state_dict'] = discriminator.state_dict()
        best_model_params['cls_state_dict'] = cls.state_dict()

        # 使用帮助函数获取优化器状态
        best_model_params['optimizer_cls_state_dict'] = get_optimizer_state(optimizer_cls)
        best_model_params['optimizer_feature_extractor_state_dict'] = get_optimizer_state(optimizer_feature_extractor)
        best_model_params['optimizer_discriminator_state_dict'] = get_optimizer_state(optimizer_discriminator)
        # 记录实验信息
        best_experiment_info['random_seed'] = random_seed
        best_experiment_info['epoch'] = best_epoch
        best_experiment_info['training_data_state'] = {
            'source_train': source_train,
            'target_train': target_train
        }

    epoch = epoch + 1

print('Best: Epoch:{}\tOS: {:.3f}\tOS*:{:.3f}\tUnk:{:.3f}\tHos:{:.3f}'.format(best_epoch, best_os, best_os_star, best_unk, best_hos))
print('class_num' + str(args.all_classes) + str(args))

# ============ Cleanup ============
print(f'\n训练结束。日志已保存至: {log_file_path}')
sys.stdout.close()
sys.stdout = orig_stdout

