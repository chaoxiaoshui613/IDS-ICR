from __future__ import division, print_function, absolute_import

import os, sys, datetime, argparse, json
import pandas as pd
import faiss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
# ============ GPU Setup ============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ============ Args ============
def get_args():
    parser = argparse.ArgumentParser(description="Open-set Domain Adaptation (CIC-IDS, no pre-filtering)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--shared_classes", type=int, default=7)
    parser.add_argument("--all_classes", type=int, default=8)
    parser.add_argument("--total_epochs", type=int, default=20)
    parser.add_argument("--log_dir", default='D:/paper_code/idea/dataset_da/log/')
    return parser.parse_args()

args = get_args()

# ============ Logging Setup ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, 'logs_cicids')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results_cicids')
for d in [LOGS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

log_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_path = os.path.join(LOGS_DIR, f'run_{log_timestamp}.log')

class Tee:
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

orig_stdout = sys.stdout
sys.stdout = Tee(log_file_path)

print('='*60)
print(f'Training Experiment (CIC-IDS, no pre-filtering)  |  {log_timestamp}')
print(f'Total epochs: {args.total_epochs}')
print(f'Log: {log_file_path}')
print('='*60)

# ============ Helpers ============
def one_hot(n_class, index):
    tmp = np.zeros((n_class,), dtype=np.float32)
    if isinstance(index, torch.Tensor):
        index = int(index.item())
    tmp[index] = 1
    return tmp

def variable_to_numpy(x):
    return x.detach().cpu().numpy()

def to_np(x):
    return x.squeeze().cpu().detach().numpy()

def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))

def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(ans)

def cal_sim(x1, x2, metric='cosine'):
    if len(x1.shape) != 2:
        x1 = x1.reshape(-1, x1.shape[-1])
    if len(x2.shape) != 2:
        x2 = x2.reshape(-1, x2.shape[-1])
    if metric == 'cosine':
        return (F.cosine_similarity(x1, x2) + 1) / 2
    return F.pairwise_distance(x1, x2) / torch.norm(x2, dim=1)

def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
    if not true_labels:
        true_labels = sorted(list(set(list(y_true))))
    true_label_to_id = {x: i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x: i for (i, x) in enumerate(pred_labels)}
    cm = np.zeros([len(true_labels), len(pred_labels)], dtype=int)
    for (true, pred) in zip(y_true, y_pred):
        cm[true_label_to_id[true]][pred_label_to_id[pred]] += 1
    return cm

def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    if label.shape != predict_prob.shape:
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

# ============ Model Classes ============
class Centroids(object):
    def __init__(self, class_num, dim, use_cuda):
        self.class_num = class_num
        self.src_ctrs = torch.ones((class_num, dim))
        self.tgt_ctrs = torch.ones((class_num, dim+1))
        self.unk_crts = torch.ones((class_num, 256))
        self.src_ctrs *= 1e-10; self.tgt_ctrs *= 1e-10; self.unk_crts *= 1e-10
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
    def update(self, pred_s, pred_t, label_s, label_unk=None):
        self.upd_src_centroids(pred_s, label_s)
        self.upd_tgt_centroids(pred_t, label_unk)

    @torch.no_grad()
    def upd_src_centroids(self, probs, labels):
        for i in range(self.class_num):
            data_idx = np.argwhere(labels[:, i] == 1)[:, 0]
            new_centroid = torch.mean(torch.tensor(probs[data_idx, :self.dim]), 0).squeeze()
            self.src_ctrs[i, :] = new_centroid

    @torch.no_grad()
    def upd_tgt_centroids(self, probs, labels):
        if labels is None: return
        for i in range(self.class_num):
            data_idx = np.argwhere(labels == i)
            new_centroid = torch.mean(torch.tensor(probs[data_idx]), 0).squeeze()
            self.tgt_ctrs[i, :] = new_centroid

class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[78, 256], normalize=True):
        super().__init__()
        self.normalize = normalize; self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        encoder_layers = []
        in_dim = input_dim
        for hd in hidden_dims:
            encoder_layers += [nn.Linear(in_dim, hd), nn.BatchNorm1d(hd), nn.ReLU()]
            in_dim = hd
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        hd_rev = hidden_dims.copy(); hd_rev.reverse()
        for i in range(len(hd_rev) - 1):
            decoder_layers += [nn.Linear(hd_rev[i], hd_rev[i+1]), nn.BatchNorm1d(hd_rev[i+1]), nn.ReLU()]
        decoder_layers.append(nn.Linear(hd_rev[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        self.__in_features = hidden_dims[-1]

    def forward(self, x):
        if self.normalize:
            if not hasattr(self, 'mean'): self.mean = torch.zeros(self.input_dim, dtype=torch.float32)
            if not hasattr(self, 'std'): self.std = torch.ones(self.input_dim, dtype=torch.float32)
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.encoder(x)

    def output_num(self):
        return self.__in_features

class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256, temp=0.05):
        super().__init__()
        self.temp = 1
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim, bias=False)
            self.main = nn.Sequential(
                self.bottleneck,
                nn.Sequential(nn.BatchNorm1d(bottle_neck_dim), nn.LeakyReLU(0.2, inplace=True), self.fc),
                nn.Softmax(dim=-1)
            )
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(self.fc, nn.Softmax(dim=-1))

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
        return out

    def virt_forward(self, K, feature_source, logits, target=None):
        if self.training and K.numel() > 0:
            if isinstance(K, np.ndarray):
                K = torch.from_numpy(K).to(torch.float32).to(feature_source.device)
            with torch.no_grad():
                W_yi = torch.gather(self.fc.weight, 0,
                    target.unsqueeze(1).expand(target.size(0), self.fc.weight.size(1)))
                W_virt = torch.norm(W_yi, dim=1).unsqueeze(-1).unsqueeze(-1) * (
                    (K / torch.norm(K, dim=1).unsqueeze(-1)).unsqueeze(0))
            vir = torch.bmm(W_virt, feature_source.unsqueeze(-1)).squeeze(-1)
            logits = torch.cat([logits, vir], dim=-1)
            return nn.Softmax(-1)(logits)
        return nn.Softmax(-1)(logits)

class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff; return input
    @staticmethod
    def backward(ctx, grad_outputs):
        return None, -ctx.coeff * grad_outputs

class GradientReverseModule(nn.Module):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler; self.global_step = 0.0; self.coeff = 0.0
        self.grl = GradientReverseLayer.apply
    def forward(self, x):
        self.coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        return self.grl(self.coeff, x)

class LargeAdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))
        self.main = nn.Sequential(
            self.ad_layer1, nn.BatchNorm1d(1024), nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer2, nn.BatchNorm1d(1024), nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer3, self.sigmoid
        )
    def forward(self, x):
        x = self.grl(x)
        for m in self.main.children():
            x = m(x)
        return x

# ============ Utility Classes ============
class Accumulator(dict):
    def __init__(self, name_or_names, accumulate_fn=np.concatenate):
        super().__init__()
        self.names = [name_or_names] if isinstance(name_or_names, str) else name_or_names
        self.accumulate_fn = accumulate_fn
        for name in self.names:
            self.__setitem__(name, [])

    def updateData(self, scope):
        for name in self.names:
            if scope[name].shape[-1] > 0:
                self.__getitem__(name).append(scope[name])

    def __enter__(self): return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb: print(exc_tb); return False
        for name in self.names:
            if len(self.__getitem__(name)) > 0:
                self.__setitem__(name, self.accumulate_fn(self.__getitem__(name)))
        return True

class TrainingModeManager:
    def __init__(self, nets, train=False):
        self.nets = nets
        self.modes = [net.training for net in nets]
        self.train = train
    def __enter__(self):
        for net in self.nets: net.train(self.train)
    def __exit__(self, *args):
        for (mode, net) in zip(self.modes, self.nets): net.train(mode)
        self.nets = None
        return True

class OptimWithSheduler:
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer; self.scheduler_func = scheduler_func; self.global_step = 0.0
        for g in self.optimizer.param_groups: g['initial_lr'] = g['lr']
    def zero_grad(self): self.optimizer.zero_grad()
    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr=g['initial_lr'])
        self.optimizer.step(); self.global_step += 1

class OptimizerManager:
    def __init__(self, optims): self.optims = optims
    def __enter__(self):
        for op in self.optims: op.zero_grad()
    def __exit__(self, *args):
        for op in self.optims: op.step()
        self.optims = None; return True

class LossCounter:
    def __init__(self):
        self.ce = 0.0; self.entropy = 0.0; self.virtual = 0.0
        self.ce_ep = 0.0; self.adv = 0.0; self.batch = 0
    def addOntBatch(self, ce, entropy, virtual, ce_ep, adv):
        self.batch += 1
        self.ce += ce.item(); self.entropy += entropy.item()
        self.virtual += virtual.item(); self.ce_ep += ce_ep.item(); self.adv += adv.item()

class DomainBus:
    def __init__(self, domainloaders):
        self.domainloaders = domainloaders
        self.domainiters = [iter(dl) for dl in domainloaders]
        self.max_iter_num = len(domainloaders[1])
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
        self.current_iter += 1; return batch_split
    def __len__(self): return self.max_iter_num
    def __iter__(self): return self
    def __next__(self):
        if self.current_iter >= self.max_iter_num:
            self.current_iter = 0; raise StopIteration
        return self.get_samples()

# ============ Data Transforms ============
def source_train_transform(data, label, is_train):
    return data, one_hot(args.all_classes, label)

def target_train_transform(data, label, is_train):
    if label in range(args.shared_classes):
        return data, one_hot(args.all_classes, label)
    return data, one_hot(args.all_classes, args.shared_classes)

_MAX_TEST_LABEL = 9
def target_test_transform(data, label, is_train):
    return data, one_hot(_MAX_TEST_LABEL, label)

def get_split_dataset_info(csv_file):
    try:
        df = pd.read_csv(csv_file)
        labels = df.iloc[:, -1].values
        data = df.iloc[:, :-1].values
        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
    except FileNotFoundError:
        print(f"File {csv_file} not found.")
        return [], []

class CustomDataset(Dataset):
    def __init__(self, data, labels, data_transformer=None):
        assert len(data) == len(labels)
        self.data = data; self.labels = labels; self.data_transformer = data_transformer
    def __len__(self): return len(self.data)
    def __getitem__(self, index, is_train=True):
        dp, lb = self.data[index], self.labels[index]
        if self.data_transformer:
            dp, lb = self.data_transformer(dp, lb, is_train)
        return dp, lb

# ============ Single Experiment Runner ============
def run_single_experiment(total_epochs=100, random_seed=42):
    torch.manual_seed(random_seed); np.random.seed(random_seed)

    print(f'\n{"="*60}')
    print(f'Total epochs: {total_epochs}')
    print(f'{"="*60}')

    # ---- Data ----
    src_csv = 'D:/paper_code/idea/dataset_da/cic/source_cicids_0_6.csv'
    tgt_csv = 'D:/paper_code/idea/dataset_da/cic/target_cicids_0_8.csv'

    src_data, src_labels = get_split_dataset_info(src_csv)
    ds = CustomDataset(src_data, src_labels, data_transformer=source_train_transform)
    source_train = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    tgt_data, tgt_labels = get_split_dataset_info(tgt_csv)
    ds1 = CustomDataset(tgt_data, tgt_labels, data_transformer=target_train_transform)
    target_train = DataLoader(ds1, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

    tgt_test_data, tgt_test_labels = get_split_dataset_info(tgt_csv)
    ds2 = CustomDataset(tgt_test_data, tgt_test_labels, data_transformer=target_test_transform)
    target_test = DataLoader(ds2, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)

    # ---- Models ----
    use_cuda = (DEVICE.type == 'cuda')
    all_centroids = Centroids(class_num=args.shared_classes, dim=args.shared_classes, use_cuda=use_cuda)
    discriminator = LargeAdversarialNetwork(256).to(DEVICE)
    feature_extractor = TabularAutoencoder(78).to(DEVICE)
    cls = CLS(feature_extractor.output_num(), args.all_classes, bottle_neck_dim=256).to(DEVICE)
    net = nn.Sequential(feature_extractor, cls).to(DEVICE)

    max_iter = 10000; warmiter = 5

    # ---- Phase 1: 初始聚类匹配（无预过滤）----
    print('\n--- Initial clustering matching ---')
    cgen = DomainBus([source_train, target_train])
    with torch.no_grad():
        with Accumulator(['fs', 'ft', 'ls', 'lt']) as Rec:
            for (dsrc, lsrc), (dtgt, ltgt) in cgen:
                _, fs, _, _ = net(dsrc.to(DEVICE))
                _, ft, _, _ = net(dtgt.to(DEVICE))
                fs, ft, ls, lt = [variable_to_numpy(x) for x in (
                    fs, ft, torch.nonzero(lsrc, as_tuple=True)[1],
                    torch.nonzero(ltgt, as_tuple=True)[1])]
                Rec.updateData(locals())

    src_feats = torch.tensor(Rec['fs'], dtype=torch.float32)
    src_labels_orig = np.array(Rec['ls'])

    # 源域所有共享类质心
    s_all_ctds = []
    for i in range(args.shared_classes):
        idx = (src_labels_orig == i)
        if idx.sum() > 0:
            s_all_ctds.append(src_feats[idx].mean(dim=0).cpu().numpy())
    s_all_ctds = np.stack(s_all_ctds, axis=0) if s_all_ctds else np.array([])

    # 目标域直接聚类
    tgt_feats = Rec['ft']
    fk = faiss.Kmeans(src_feats.size(1), args.all_classes, niter=800, verbose=False, min_points_per_centroid=1, gpu=False)
    fk.train(tgt_feats)
    t_full = fk.centroids

    # 匈牙利匹配
    if len(s_all_ctds) > 0 and len(t_full) > 0:
        cost = np.linalg.norm(s_all_ctds[:, None, :] - t_full[None, :, :], axis=-1)
        _, t_match = linear_sum_assignment(cost)
        nomatch_init = [t_full[i] for i in range(len(t_full)) if i not in t_match]
        nomatch = np.stack(nomatch_init, axis=0) if nomatch_init else np.array([])
    else:
        nomatch = np.array([])
    print(f'Source centroids: {len(s_all_ctds)}, Target centroids: {len(t_full)}, Unmatched (virtual): {len(nomatch)}')

    del Rec

    # ---- Optimizers ----
    sched = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=max_iter)
    opt_d = OptimWithSheduler(optim.SGD(discriminator.parameters(), lr=args.learning_rate*10, weight_decay=5e-4, momentum=0.9, nesterov=True), sched)
    opt_f = OptimWithSheduler(optim.SGD(feature_extractor.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True), sched)
    opt_c = OptimWithSheduler(optim.SGD(cls.parameters(), lr=args.learning_rate*10, weight_decay=5e-4, momentum=0.9, nesterov=True), sched)

    # ---- Training ----
    epoch = 0; best_hos = 0; best_unk = 0; best_os_star = 0
    hos_hist, unk_hist, os_star_hist, acc_hist = [], [], [], []

    if len(nomatch) == 0:
        print('WARNING: nomatch empty, using random init')
        nomatch = np.random.randn(1, 256).astype(np.float32)
        nomatch = nomatch / np.linalg.norm(nomatch, axis=-1, keepdims=True)

    while epoch < total_epochs:
        cgen = DomainBus([source_train, target_train])
        lc = LossCounter()

        with Accumulator(['pred_s', 'pred_t', 'label_s', 'kl', 'fss', 'ftt']) as Rec:
            for (im_s, label_s), (im_t, label_t) in cgen:
                im_s = im_s.to(DEVICE); label_s = label_s.to(DEVICE)
                im_t = im_t.to(DEVICE); label_t = label_t.to(DEVICE)

                _, feat_s, fc_s, pred_s = net(im_s)
                ft1, feat_t, fc_t, pred_t = net(im_t)

                d_prob_s = discriminator(feat_s); d_prob_t = discriminator(feat_t)

                # KL divergence
                s_ctds, _ = all_centroids.get_centroids()
                _, pseudo_t_lbl = pred_t[:, :args.shared_classes].max(1)
                klt = F.kl_div(
                    nn.Softmax(-1)(fc_t[:, :args.shared_classes]).log(),
                    s_ctds[pseudo_t_lbl], reduction='none'
                ).sum(1).detach()
                klt = torch.where(torch.isinf(klt), torch.full_like(klt, 10.0), klt)

                # GMM
                klt_np = to_np(klt)[:, None]
                gmm = GaussianMixture(n_components=2, covariance_type='full', n_init=1).fit(klt_np)
                kn_cluster = np.argmin(gmm.means_); un_cluster = np.argmax(gmm.means_)
                gmm_idx = gmm.predict(klt_np)

                # 保存tensor引用，避免被numpy覆盖后报错
                label_s_t, label_t_t = label_s, label_t
                pred_t_t, feat_s_t = pred_t, feat_s  # 保留tensor版本
                _pred_s, _pred_t, _label_s, _kl, _fss, _ftt = [
                    variable_to_numpy(x) for x in (
                        nn.Softmax(-1)(fc_s[:, :args.shared_classes]),
                        pred_t, label_s, klt, feat_s, feat_t
                    )]
                pred_s, pred_t, label_s, kl, fss, ftt = _pred_s, _pred_t, _label_s, _kl, _fss, _ftt
                Rec.updateData(locals())

                weight = gmm.predict_proba(klt_np)[:, kn_cluster]
                weight = torch.tensor(weight, device=DEVICE).detach()

                if epoch <= 10:
                    weight = torch.where(weight > 0.8, torch.tensor(1.0, device=DEVICE), torch.tensor(0.0, device=DEVICE)).detach()
                    r = torch.nonzero(torch.tensor(gmm_idx != kn_cluster, device=DEVICE)).squeeze(-1)
                    if r.size(0) > 16:
                        r = torch.sort(klt.detach(), dim=0)[1][-16:]
                else:
                    weight = torch.where(torch.tensor(gmm_idx == kn_cluster, device=DEVICE), torch.tensor(1.0, device=DEVICE), torch.tensor(0.0, device=DEVICE)).detach()
                    r = torch.nonzero(torch.tensor(gmm_idx == un_cluster, device=DEVICE)).squeeze(-1)

                # L_unk
                if r.numel() == 0:
                    ce_ep = torch.tensor(0.0, device=DEVICE)
                else:
                    r = r.view(-1)
                    feat_unk = torch.index_select(ft1, 0, r)
                    if r.size(0) == 1:
                        with torch.no_grad():
                            cls.eval()
                            _, feat_unk, logits_unk, pred_unk = cls(feat_unk)
                            cls.train()
                    else:
                        _, feat_unk, logits_unk, pred_unk = cls(feat_unk)
                    _, pseudo_idx = pred_unk[:, args.shared_classes:].max(1)
                    pseudo_idx = pseudo_idx + args.shared_classes
                    pseudo_label = torch.zeros(r.size(0), args.all_classes, device=DEVICE)
                    pseudo_label = pseudo_label.scatter_(1, pseudo_idx.unsqueeze(1), 1.0)
                    ce_ep = CrossEntropyLoss(pseudo_label, pred_unk)

                # Losses
                ce = CrossEntropyLoss(label_s_t, nn.Softmax(-1)(fc_s))

                nm = torch.from_numpy(nomatch).to(DEVICE) if isinstance(nomatch, np.ndarray) else nomatch.to(DEVICE)
                if nm.numel() > 0:
                    virt_pred = cls.virt_forward(nm, feat_s_t, fc_s[:, :], torch.nonzero(label_s_t)[:, 1])
                    pz = torch.zeros(label_s_t.size(0), nm.size(0), device=DEVICE)
                    v_label = torch.cat((label_s_t[:, :], pz), 1)
                    virtual_ce = CrossEntropyLoss(v_label, virt_pred)
                else:
                    virtual_ce = torch.tensor(0.0, device=DEVICE)

                entropy = EntropyLoss(pred_t_t[:, :], instance_level_weight=weight.contiguous())
                adv_loss = BCELossForMultiClassification(label=torch.ones_like(d_prob_s), predict_prob=d_prob_s)
                adv_loss += BCELossForMultiClassification(label=torch.ones_like(d_prob_t), predict_prob=1-d_prob_t, instance_level_weight=weight)

                with OptimizerManager([opt_c, opt_f, opt_d]):
                    if epoch <= warmiter:
                        loss = ce + 0.3*virtual_ce + 0.4*adv_loss + 0.4*entropy + 0*ce_ep
                    else:
                        loss = ce + 0.5*virtual_ce + 0.5*adv_loss + 0.5*entropy + 1*ce_ep
                    loss.backward()

                lc.addOntBatch(ce, entropy, virtual_ce, ce_ep, adv_loss)

        # ---- End of epoch ----
        all_centroids.update(Rec['pred_s'], Rec['pred_t'], Rec['label_s'])

        s_ctds = []
        for i in range(args.shared_classes):
            mask = (np.nonzero(Rec['label_s'])[1] == i)
            if mask.sum() > 0:
                s_ctds.append(Rec['fss'][mask].mean(axis=0))
            else:
                s_ctds.append(np.random.randn(Rec['fss'].shape[1]).astype(np.float32))
        s_ctds = np.stack(s_ctds, axis=0)

        # ===== Clustering & Matching (无预过滤，直接聚类) =====
        fk = faiss.Kmeans(256, args.all_classes, niter=800, verbose=False, min_points_per_centroid=1, gpu=False)
        fk.train(Rec['ftt'])
        t_full = fk.centroids
        cost = np.linalg.norm(s_ctds[:, None, :] - t_full[None, :, :], axis=-1)
        _, tm = linear_sum_assignment(cost)
        nomatch = np.stack([t_full[i] for i in range(args.all_classes) if i not in tm], axis=0) if args.all_classes > len(tm) else np.array([])

        if len(nomatch) == 0:
            nomatch = np.random.randn(1, 256).astype(np.float32)
            nomatch = nomatch / np.linalg.norm(nomatch, axis=-1, keepdims=True)

        # ---- Unknown class weight init ----
        if epoch == warmiter:
            fk = faiss.Kmeans(256, args.all_classes, niter=800, verbose=False, min_points_per_centroid=1, gpu=False)
            fk.train(Rec['ftt'])
            t_init = fk.centroids
            cost = np.linalg.norm(s_ctds[:, None, :] - t_init[None, :, :], axis=-1)
            _, tm_init = linear_sum_assignment(cost)
            init_uw = np.stack([t_init[i] for i in range(args.all_classes) if i not in tm_init], axis=0)
            for key, v in net.state_dict().items():
                if key == '1.main.1.2.weight':
                    v.requires_grad = False
                    net.state_dict()['1.fc.weight'].requires_grad = False
                    vvnorm = torch.norm(v, dim=-1).mean().cpu().numpy()
                    init_uw = init_uw / np.linalg.norm(init_uw, axis=-1, keepdims=True) * vvnorm
                    fcnew = np.concatenate([v[:args.shared_classes].clone().detach().cpu().numpy(), init_uw], axis=0)
                    net.state_dict()['1.fc.weight'].copy_(torch.from_numpy(fcnew).to(DEVICE))
                    v.requires_grad = True
                    net.state_dict()['1.fc.weight'].requires_grad = True

        # ---- Evaluation ----
        eval_gmm = BayesianGaussianMixture(n_components=2, max_iter=800).fit(Rec['kl'][:, None])

        with TrainingModeManager([feature_extractor, cls], train=False):
            with Accumulator(['predict_prob', 'predict_index', 'label']) as acc:
                for im, label in target_test:
                    _, _, _, predict_prob = net(im.to(DEVICE))
                    predict_prob, label = [variable_to_numpy(x) for x in (predict_prob, label)]
                    label = np.argmax(label, axis=-1).reshape(-1, 1)
                    predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
                    acc.updateData(locals())

        predict_prob = acc['predict_prob']
        predict_index = acc['predict_index']
        label = acc['label']

        y_true = label.flatten(); y_pred = predict_index.flatten()
        yy_max = int(y_true.max())
        m = extended_confusion_matrix(y_true, y_pred, true_labels=list(range(yy_max+1)), pred_labels=list(range(args.all_classes)))
        m_m = np.copy(m)
        n_ur = m_m.shape[0] - args.shared_classes
        if n_ur > 1:
            m_m[args.shared_classes, :] = m_m[args.shared_classes:, :].sum(axis=0)
            m_m = m_m[:args.shared_classes+1, :]

        cm = m_m.astype(float)
        rs = np.sum(cm, axis=1, keepdims=True); rs[rs == 0] = 1
        cm /= rs
        os_star = sum([cm[i][i] for i in range(args.shared_classes)]) / args.shared_classes
        unk_row = cm[-1:]
        cu = unk_row[:, -1].sum(); tu = unk_row.sum()
        unkn = cu / tu if tu > 0 else 0.0
        hos = (2 * os_star * unkn) / (os_star + unkn) if (os_star + unkn) != 0 else 0.0
        acc_o = (os_star * args.shared_classes + unkn) / (args.shared_classes + 1)

        hos_hist.append(hos); unk_hist.append(unkn); os_star_hist.append(os_star); acc_hist.append(acc_o)

        c_avg = lc.ce / max(lc.batch,1); e_avg = lc.entropy / max(lc.batch,1)
        v_avg = lc.virtual / max(lc.batch,1); ep_avg = lc.ce_ep / max(lc.batch,1); a_avg = lc.adv / max(lc.batch,1)
        for vv in [c_avg, e_avg, v_avg, ep_avg, a_avg]:
            if np.isnan(vv): vv = 0.0

        print(f'E{epoch:3d} | OS:{acc_o:.3f} OS*:{os_star:.3f} UNK:{unkn:.3f} HOS:{hos:.3f} | ce:{c_avg:.2f} ent:{e_avg:.2f} virt:{v_avg:.2f} ep:{ep_avg:.2f} adv:{a_avg:.2f}')
        print(f'  Confusion Matrix:\n{np.round(cm, 3)}')

        if hos > best_hos: best_hos = hos; best_unk = unkn; best_os_star = os_star
        epoch += 1

    print(f'\nDONE | Best: HOS={best_hos:.4f} UNK={best_unk:.4f} OS*={best_os_star:.4f}')
    return {'hos': hos_hist, 'unk': unk_hist, 'os_star': os_star_hist,
            'accuracy': acc_hist, 'best_hos': best_hos, 'best_unk': best_unk, 'best_os_star': best_os_star}

# ============ MAIN ============
if __name__ == '__main__':
    result = run_single_experiment(total_epochs=args.total_epochs)

    # ---- Save summary ----
    summary = {
        'best_hos': float(result['best_hos']),
        'best_unk': float(result['best_unk']),
        'best_os_star': float(result['best_os_star'])
    }
    with open(os.path.join(RESULTS_DIR, f'summary_{log_timestamp}.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- Plot curves ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].plot(result['hos'], color='steelblue', alpha=0.8)
    axes[0,1].plot(result['unk'], color='steelblue', alpha=0.8)
    axes[1,0].plot(result['os_star'], color='steelblue', alpha=0.8)
    axes[1,1].plot(result['accuracy'], color='steelblue', alpha=0.8)

    axes[0,0].set_title('HOS (Harmonic Mean)'); axes[0,0].set_xlabel('Epoch')
    axes[0,1].set_title('UNK (Unknown Accuracy)'); axes[0,1].set_xlabel('Epoch')
    axes[1,0].set_title('OS* (Known Accuracy)'); axes[1,0].set_xlabel('Epoch')
    axes[1,1].set_title('Overall Accuracy'); axes[1,1].set_xlabel('Epoch')
    plt.suptitle(f'Training Curves - CIC-IDS ({args.total_epochs} epochs)', fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f'curves_{log_timestamp}.png')
    plt.savefig(fig_path, dpi=150)
    print(f'\nCurves saved: {fig_path}')

    # ---- Summary ----
    print(f'\n{"="*60}')
    print(f'Best HOS: {result["best_hos"]:.4f}')
    print(f'Best UNK: {result["best_unk"]:.4f}')
    print(f'Best OS*: {result["best_os_star"]:.4f}')
    print('='*60)

    sys.stdout.close(); sys.stdout = orig_stdout
    print(f'\nDone. Results: {RESULTS_DIR}  |  Log: {log_file_path}')
