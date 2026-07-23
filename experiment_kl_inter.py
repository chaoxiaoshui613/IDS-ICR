
from __future__ import division, print_function, absolute_import

import os, sys, datetime, argparse, json
import pandas as pd
import faiss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix, roc_curve, auc as sk_auc, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_args():
    parser = argparse.ArgumentParser(description="Reviewer Response Analysis (CIC-IDS, no pre-filtering)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--shared_classes", type=int, default=7)
    parser.add_argument("--all_classes", type=int, default=8)
    parser.add_argument("--total_epochs", type=int, default=20)
    parser.add_argument("--fixed_quantile", type=float, default=0.7, help="固定阈值分位数")
    return parser.parse_args()

args = get_args()
BASE_DIR = os.path.join(SCRIPT_DIR, 'reviewer_response')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_path = os.path.join(LOG_DIR, f'response_{timestamp}.log')

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

print('='*70)
print(f'Reviewer Response Analysis (CIC-IDS, no pre-filtering)  |  {timestamp}')
print(f'Total epochs: {args.total_epochs}')
print(f'Log: {log_file_path}')
print('='*70)

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

class Accumulator(object):
    def __init__(self, keys):
        self.keys = keys
        self.data = {k: [] for k in keys}
    def updateData(self, d):
        for k in self.keys:
            if k in d:
                self.data[k].append(d[k])
    def __enter__(self):
        return self
    def __exit__(self, *args):
        for k in self.keys:
            if self.data[k]:
                if isinstance(self.data[k][0], np.ndarray):
                    self.data[k] = np.concatenate(self.data[k], axis=0)
                elif isinstance(self.data[k][0], torch.Tensor):
                    self.data[k] = torch.cat(self.data[k], dim=0)

class OptimWithSheduler(torch.optim.Optimizer):
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.step_num = 0
    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scheduler_func(self.step_num, param_group.get('initial_lr', param_group['lr']))
        self.optimizer.step()
        self.step_num += 1
    def zero_grad(self):
        self.optimizer.zero_grad()
    def state_dict(self):
        return self.optimizer.state_dict()

class LossCounter:
    def __init__(self):
        self.ce = 0; self.entropy = 0; self.virtual = 0; self.ce_ep = 0; self.adv = 0; self.batch = 0
    def addOntBatch(self, ce, entropy, virtual, ce_ep, adv):
        self.ce += float(ce.item()); self.entropy += float(entropy.item())
        self.virtual += float(virtual.item()); self.ce_ep += float(ce_ep.item())
        self.adv += float(adv.item()); self.batch += 1

class OptimizerManager:
    def __init__(self, optimizers):
        self.optimizers = optimizers
    def __enter__(self):
        for op in self.optimizers:
            op.zero_grad()
        return self
    def __exit__(self, type, value, traceback):
        for op in self.optimizers:
            op.step()

class TrainingModeManager:
    def __init__(self, models, train=True):
        self.models = models
        self.train = train
        self.prev_modes = [m.training for m in models]
    def __enter__(self):
        for m in self.models:
            m.train(self.train)
        return self
    def __exit__(self, *args):
        for m, prev in zip(self.models, self.prev_modes):
            m.train(prev)

class CustomDataset(Dataset):
    def __init__(self, data, labels, data_transformer=None):
        self.data = data
        self.labels = labels
        self.data_transformer = data_transformer
    def __len__(self): return len(self.data)
    def __getitem__(self, index, is_train=True):
        dp, lb = self.data[index], self.labels[index]
        if self.data_transformer:
            dp, lb = self.data_transformer(dp, lb, is_train)
        return dp, lb

def get_split_dataset_info(filepath):
    data = []; labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            label = int(parts[-1])
            feat = list(map(float, parts[:-1]))
            data.append(feat)
            labels.append(label)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64)

def to_categorical(y, num_classes=None):
    num_classes = num_classes or np.max(y) + 1
    return np.eye(num_classes)[y]

source_train_transform = lambda x, y, is_train: (torch.from_numpy(x).float(), torch.from_numpy(to_categorical(y, args.shared_classes)).float())

def target_train_transform(x, y, is_train):
    if y in range(args.shared_classes):
        lbl = y
    else:
        lbl = args.shared_classes
    return torch.from_numpy(x).float(), torch.from_numpy(to_categorical(lbl, args.all_classes)).float()

def target_test_transform(x, y, is_train):
    return torch.from_numpy(x).float(), torch.from_numpy(to_categorical(y, 9)).float()

class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(TabularAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = [input_dim, 256]
        self.fc1 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        return h
    def output_num(self):
        return self.hidden_dims[-1]

class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, bottle_neck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottle_neck_dim, out_dim)
        )
        self.fc = self.main[2]
        self.bottle = self.main[0]
    def forward(self, x):
        feat = self.bottle(x)
        out = self.fc(feat)
        return x, feat, out, F.softmax(out, dim=1)
    def virt_forward(self, virt_w, feat, fc_s, label_s):
        virt_w = virt_w.view(-1, virt_w.size(-1))
        w_norm = torch.norm(self.fc.weight, dim=-1).mean()
        virt_w_norm = torch.norm(virt_w, dim=-1, keepdim=True)
        virt_w = virt_w / virt_w_norm * w_norm
        feat_norm = torch.norm(feat, dim=-1, keepdim=True)
        feat_unit = feat / feat_norm
        virt_sim = torch.mm(feat_unit, virt_w.t())
        fc_s_norm = torch.norm(fc_s, dim=-1, keepdim=True)
        fc_s_unit = fc_s / fc_s_norm
        known_sim = torch.mm(fc_s_unit, self.fc.weight[:args.shared_classes].t())
        combined = torch.cat([known_sim, virt_sim], dim=-1)
        return combined

class LargeAdversarialNetwork(nn.Module):
    def __init__(self, input_dim):
        super(LargeAdversarialNetwork, self).__init__()
        layers = [nn.Linear(input_dim, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
                  nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
                  nn.Linear(1024, 1)]
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return torch.sigmoid(self.main(x))

class Centroids:
    def __init__(self, class_num, dim, use_cuda=True):
        self.class_num = class_num
        self.dim = dim
        self.use_cuda = use_cuda
        self.centroids = np.zeros((class_num, dim))
        self.counter = np.zeros(class_num)
    def update(self, pred_s, pred_t, label_s):
        for i in range(self.class_num):
            mask = (np.nonzero(label_s)[1] == i)
            if mask.sum() > 0:
                self.centroids[i] = pred_s[mask].mean(axis=0)
                self.counter[i] += 1
    def get_centroids(self):
        return torch.from_numpy(self.centroids).float().to(DEVICE) if self.use_cuda else torch.from_numpy(self.centroids).float(), self.counter

def CrossEntropyLoss(label, predict_prob, reduction='mean'):
    return -(label * predict_prob.log()).sum(dim=1).mean() if reduction == 'mean' else -(label * predict_prob.log()).sum(dim=1)

def BCELossForMultiClassification(label, predict_prob, reduction='mean'):
    eps = 1e-8
    predict_prob = torch.clamp(predict_prob, eps, 1 - eps)
    return -(label * predict_prob.log() + (1 - label) * (1 - predict_prob).log()).mean() if reduction == 'mean' else -(label * predict_prob.log() + (1 - label) * (1 - predict_prob).log())

def EntropyLoss(input_, reduction='mean', instance_level_weight=None):
    eps = 1e-8
    input_ = torch.clamp(input_, eps, 1 - eps)
    entropy = -input_ * torch.log(input_)
    entropy = entropy.sum(dim=1)
    if instance_level_weight is not None:
        entropy = entropy * instance_level_weight
    return entropy.mean() if reduction == 'mean' else entropy

def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
    if true_labels is None:
        true_labels = sorted(np.unique(y_true))
    if pred_labels is None:
        pred_labels = sorted(np.unique(y_pred))
    true_to_idx = {l: i for i, l in enumerate(true_labels)}
    pred_to_idx = {l: i for i, l in enumerate(pred_labels)}
    m = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in true_to_idx and p in pred_to_idx:
            m[true_to_idx[t], pred_to_idx[p]] += 1
    return m

class DomainBus:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.iterators = [iter(d) for d in dataloaders]
    def __iter__(self):
        return self
    def __next__(self):
        results = []
        for i, it in enumerate(self.iterators):
            try:
                results.append(next(it))
            except StopIteration:
                self.iterators[i] = iter(self.dataloaders[i])
                results.append(next(self.iterators[i]))
        return results

def compute_feature_metrics(features, labels, n_known_classes):
    binary_labels = np.where(labels >= n_known_classes, n_known_classes, labels)
    unique_labels = np.unique(binary_labels)
    
    centers = {}
    for c in unique_labels:
        centers[c] = features[binary_labels == c].mean(axis=0)
    
    intra_distances = {}
    for c in unique_labels:
        class_feats = features[binary_labels == c]
        if len(class_feats) > 1:
            dists = pairwise_distances(class_feats, class_feats, metric='euclidean')
            intra_distances[c] = dists[np.triu_indices_from(dists, k=1)].mean()
        else:
            intra_distances[c] = 0.0
    
    inter_distances = {}
    class_list = list(centers.keys())
    for i in range(len(class_list)):
        for j in range(i + 1, len(class_list)):
            c1, c2 = class_list[i], class_list[j]
            dist = np.linalg.norm(centers[c1] - centers[c2])
            inter_distances[f'{c1}-{c2}'] = dist
    
    known_intra = np.mean([intra_distances[c] for c in range(n_known_classes) if c in intra_distances])
    unk_intra = intra_distances.get(n_known_classes, 0.0)
    known_unk_inter = np.mean([inter_distances[f'{c}-{n_known_classes}']
                               for c in range(n_known_classes)
                               if f'{c}-{n_known_classes}' in inter_distances])
    known_known_inter = []
    for i in range(n_known_classes):
        for j in range(i + 1, n_known_classes):
            key = f'{i}-{j}'
            if key in inter_distances:
                known_known_inter.append(inter_distances[key])
    known_known_inter = np.mean(known_known_inter) if known_known_inter else 0.0
    
    sil_score = silhouette_score(features, binary_labels, metric='euclidean') if len(unique_labels) >= 2 else 0.0
    
    return {
        'silhouette': float(sil_score),
        'known_intra': float(known_intra),
        'unknown_intra': float(unk_intra),
        'known_known_inter': float(known_known_inter),
        'known_unknown_inter': float(known_unk_inter),
        'separation_ratio': float(known_unk_inter / max(known_intra, 1e-6)),
        'intra_per_class': {int(k): float(v) for k, v in intra_distances.items()},
        'inter_per_pair': inter_distances,
    }

def run_single_experiment(total_epochs=100, random_seed=42):
    torch.manual_seed(random_seed); np.random.seed(random_seed)

    print(f'\n{"="*70}')
    print(f'Total epochs: {total_epochs}')
    print(f'{"="*70}')

    # ===== 修改这里的路径 =====
    src_csv = 'D:/paper_code/idea/dataset_da/cic/source_cicids_0_6.csv'
    tgt_csv = 'D:/paper_code/idea/dataset_da/cic/target_cicids_0_8.csv'

    src_data, src_labels = get_split_dataset_info(src_csv)
    ds = CustomDataset(src_data, src_labels, data_transformer=source_train_transform)
    source_train = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    tgt_data, tgt_labels = get_split_dataset_info(tgt_csv)
    ds1 = CustomDataset(tgt_data, tgt_labels, data_transformer=target_train_transform)
    target_train = DataLoader(ds1, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

    ds2 = CustomDataset(tgt_data, tgt_labels, data_transformer=target_test_transform)
    target_test = DataLoader(ds2, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=False)

    use_cuda = (DEVICE.type == 'cuda')
    all_centroids = Centroids(class_num=args.shared_classes, dim=args.shared_classes, use_cuda=use_cuda)
    discriminator = LargeAdversarialNetwork(256).to(DEVICE)
    feature_extractor = TabularAutoencoder(78).to(DEVICE)
    cls = CLS(feature_extractor.output_num(), args.all_classes, bottle_neck_dim=256).to(DEVICE)
    net = nn.Sequential(feature_extractor, cls).to(DEVICE)

    max_iter = 10000; warmiter = 5

    print('\n--- Initial clustering matching (no pre-filtering) ---')
    # 采样计算，避免CPU内存溢出
    max_samples = 20000
    src_feat_list, src_lbl_list = [], []
    tgt_feat_list = []
    src_count, tgt_count = 0, 0
    cgen = DomainBus([source_train, target_train])
    with torch.no_grad():
        for (dsrc, lsrc), (dtgt, ltgt) in cgen:
            if src_count < max_samples:
                _, fs, _, _ = net(dsrc.to(DEVICE))
                src_feat_list.append(variable_to_numpy(fs))
                src_lbl_list.append(torch.nonzero(lsrc, as_tuple=True)[1].cpu().numpy())
                src_count += len(fs)
                del fs
            if tgt_count < max_samples:
                _, ft, _, _ = net(dtgt.to(DEVICE))
                tgt_feat_list.append(variable_to_numpy(ft))
                tgt_count += len(ft)
                del ft
            del dsrc, dtgt, lsrc, ltgt
            if src_count >= max_samples and tgt_count >= max_samples:
                break

    src_feats_np = np.concatenate(src_feat_list, axis=0)
    src_labels_orig = np.concatenate(src_lbl_list, axis=0).flatten()
    tgt_feats_np = np.concatenate(tgt_feat_list, axis=0)
    del src_feat_list, src_lbl_list, tgt_feat_list

    s_all_ctds = []
    for i in range(args.shared_classes):
        idx = (src_labels_orig == i)
        if idx.sum() > 0:
            s_all_ctds.append(src_feats_np[idx].mean(axis=0))
    s_all_ctds = np.stack(s_all_ctds, axis=0) if s_all_ctds else np.array([])

    fk = faiss.Kmeans(src_feats_np.shape[1], args.all_classes, niter=800, verbose=False, min_points_per_centroid=1, gpu=False)
    fk.train(tgt_feats_np)
    t_full = fk.centroids

    if len(s_all_ctds) > 0 and len(t_full) > 0:
        cost = np.linalg.norm(s_all_ctds[:, None, :] - t_full[None, :, :], axis=-1)
        _, t_match = linear_sum_assignment(cost)
        nomatch_init = [t_full[i] for i in range(len(t_full)) if i not in t_match]
        nomatch = np.stack(nomatch_init, axis=0) if nomatch_init else np.array([])
    else:
        nomatch = np.array([])
    print(f'Source centroids: {len(s_all_ctds)}, Target centroids: {len(t_full)}, Unmatched (virtual): {len(nomatch)}')

    del Rec

    sched = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=max_iter)
    opt_d = OptimWithSheduler(optim.SGD(discriminator.parameters(), lr=args.learning_rate*10, weight_decay=5e-4, momentum=0.9, nesterov=True), sched)
    opt_f = OptimWithSheduler(optim.SGD(feature_extractor.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True), sched)
    opt_c = OptimWithSheduler(optim.SGD(cls.parameters(), lr=args.learning_rate*10, weight_decay=5e-4, momentum=0.9, nesterov=True), sched)

    epoch = 0; best_hos = 0; best_epoch = 0
    hos_hist, unk_hist, os_star_hist, acc_hist = [], [], [], []

    if len(nomatch) == 0:
        print('WARNING: nomatch empty, using random init')
        nomatch = np.random.randn(1, 256).astype(np.float32)
        nomatch = nomatch / np.linalg.norm(nomatch, axis=-1, keepdims=True)

    best_features = None
    best_labels = None
    best_kl_values = None

    while epoch < total_epochs:
        cgen = DomainBus([source_train, target_train])
        lc = LossCounter()
        max_train_samples = 20000
        src_cnt, tgt_cnt = 0, 0

        with Accumulator(['pred_s', 'pred_t', 'label_s', 'kl', 'fss', 'ftt']) as Rec:
            for (im_s, label_s), (im_t, label_t) in cgen:
                im_s = im_s.to(DEVICE); label_s = label_s.to(DEVICE)
                im_t = im_t.to(DEVICE); label_t = label_t.to(DEVICE)

                _, feat_s, fc_s, pred_s = net(im_s)
                ft1, feat_t, fc_t, pred_t = net(im_t)

                d_prob_s = discriminator(feat_s); d_prob_t = discriminator(feat_t)

                s_ctds, _ = all_centroids.get_centroids()
                _, pseudo_t_lbl = pred_t[:, :args.shared_classes].max(1)
                klt = F.kl_div(
                    nn.Softmax(-1)(fc_t[:, :args.shared_classes]).log(),
                    s_ctds[pseudo_t_lbl], reduction='none'
                ).sum(1).detach()
                klt = torch.where(torch.isinf(klt), torch.full_like(klt, 10.0), klt)

                klt_np = to_np(klt)[:, None]
                gmm = GaussianMixture(n_components=2, covariance_type='full', n_init=1).fit(klt_np)
                kn_cluster = np.argmin(gmm.means_); un_cluster = np.argmax(gmm.means_)
                gmm_idx = gmm.predict(klt_np)

                label_s_t, label_t_t = label_s, label_t
                pred_t_t, feat_s_t = pred_t, feat_s
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

        all_centroids.update(Rec['pred_s'], Rec['pred_t'], Rec['label_s'])

        s_ctds = []
        for i in range(args.shared_classes):
            mask = (np.nonzero(Rec['label_s'])[1] == i)
            if mask.sum() > 0:
                s_ctds.append(Rec['fss'][mask].mean(axis=0))
            else:
                s_ctds.append(np.random.randn(Rec['fss'].shape[1]).astype(np.float32))
        s_ctds = np.stack(s_ctds, axis=0)

        fk = faiss.Kmeans(256, args.all_classes, niter=800, verbose=False, min_points_per_centroid=1, gpu=False)
        fk.train(Rec['ftt'])
        t_full = fk.centroids
        cost = np.linalg.norm(s_ctds[:, None, :] - t_full[None, :, :], axis=-1)
        _, tm = linear_sum_assignment(cost)
        nomatch = np.stack([t_full[i] for i in range(args.all_classes) if i not in tm], axis=0) if args.all_classes > len(tm) else np.array([])

        if len(nomatch) == 0:
            nomatch = np.random.randn(1, 256).astype(np.float32)
            nomatch = nomatch / np.linalg.norm(nomatch, axis=-1, keepdims=True)

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

        # ---- Evaluation with KL and features ----
        eval_gmm = BayesianGaussianMixture(n_components=2, max_iter=800).fit(Rec['kl'][:, None])

        with TrainingModeManager([feature_extractor, cls], train=False):
            with Accumulator(['predict_prob', 'predict_index', 'label', 'feat_test', 'kl_test']) as acc:
                for im, label in target_test:
                    _, feat, fc, predict_prob = net(im.to(DEVICE))
                    
                    true_l = np.argmax(label.numpy(), axis=-1).reshape(-1, 1)
                    predict_index = np.argmax(variable_to_numpy(predict_prob), axis=-1).reshape(-1, 1)
                    
                    s_ctds_eval, _ = all_centroids.get_centroids()
                    _, pseudo_lbl_eval = predict_prob[:, :args.shared_classes].max(1)
                    klt_eval = F.kl_div(
                        nn.Softmax(-1)(fc[:, :args.shared_classes]).log(),
                        s_ctds_eval[pseudo_lbl_eval], reduction='none'
                    ).sum(1).detach()
                    klt_eval = torch.where(torch.isinf(klt_eval), torch.full_like(klt_eval, 10.0), klt_eval)
                    
                    acc.updateData({
                        'predict_prob': variable_to_numpy(predict_prob),
                        'predict_index': predict_index,
                        'label': true_l,
                        'feat_test': variable_to_numpy(feat),
                        'kl_test': variable_to_numpy(klt_eval)
                    })

        predict_prob = acc['predict_prob']
        predict_index = acc['predict_index']
        label = acc['label']
        test_features = acc['feat_test']
        kl_test = acc['kl_test']

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

        if hos > best_hos:
            best_hos = hos; best_epoch = epoch
            best_features = test_features
            best_labels = y_true
            best_kl_values = kl_test

            torch.save({
                'net': net.state_dict(),
                'epoch': epoch,
                'metrics': {'hos': hos, 'unk': unkn, 'os_star': os_star},
            }, os.path.join(DATA_DIR, f'best_model_{timestamp}.pth'))

        epoch += 1

    print(f'\nDONE | Best: Epoch={best_epoch} HOS={best_hos:.4f}')
    
    np.savez(os.path.join(DATA_DIR, f'best_features_{timestamp}.npz'),
             features=best_features, labels=best_labels, kl_values=best_kl_values)
    
    return {
        'hos': hos_hist, 'unk': unk_hist, 'os_star': os_star_hist,
        'accuracy': acc_hist, 'best_hos': best_hos, 'best_epoch': best_epoch,
        'features': best_features, 'labels': best_labels, 'kl_values': best_kl_values
    }

def analyze_and_plot(result):
    print('\n' + '='*70)
    print('Reviewer Response Analysis')
    print('='*70)
    
    features = result['features']
    labels = result['labels']
    kl_values = result['kl_values']
    
    is_unknown = (labels >= args.shared_classes).astype(int)
    kl_known = kl_values[is_unknown == 0]
    kl_unknown = kl_values[is_unknown == 1]
    
    print(f'\nUnknown detection stats:')
    print(f'  Known samples: {len(kl_known)}, Unknown samples: {len(kl_unknown)}')
    print(f'  KL - Known mean: {kl_known.mean():.4f}, Unknown mean: {kl_unknown.mean():.4f}')
    
    # ===== Plot 1: KL Distribution + AUROC (GMM vs Fixed Threshold) =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    bins = np.linspace(kl_values.min(), np.percentile(kl_values, 99), 50)
    axes[0].hist(kl_known, bins=bins, alpha=0.6, label=f'Known (n={len(kl_known)})', color='steelblue', density=True)
    axes[0].hist(kl_unknown, bins=bins, alpha=0.6, label=f'Unknown (n={len(kl_unknown)})', color='coral', density=True)
    
    klt_np = kl_values[:, None]
    gmm = GaussianMixture(n_components=2, covariance_type='full', n_init=1).fit(klt_np)
    gmm_means = gmm.means_.flatten()
    kn_cluster = np.argmin(gmm_means); un_cluster = np.argmax(gmm_means)
    axes[0].axvline(gmm_means[kn_cluster], color='steelblue', linestyle='--', alpha=0.8, label=f'GMM Known μ={gmm_means[kn_cluster]:.2f}')
    axes[0].axvline(gmm_means[un_cluster], color='coral', linestyle='--', alpha=0.8, label=f'GMM Unknown μ={gmm_means[un_cluster]:.2f}')
    
    fixed_threshold = np.quantile(kl_values, args.fixed_quantile)
    axes[0].axvline(fixed_threshold, color='green', linestyle=':', alpha=0.8, label=f'Fixed q={args.fixed_quantile}')
    
    axes[0].set_xlabel('KL Divergence')
    axes[0].set_ylabel('Density')
    axes[0].set_title('KL Divergence Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    fpr_kl, tpr_kl, _ = roc_curve(is_unknown, kl_values, pos_label=1)
    auroc_kl = sk_auc(fpr_kl, tpr_kl)
    
    gmm_probs = gmm.predict_proba(klt_np)[:, un_cluster]
    fpr_gmm, tpr_gmm, _ = roc_curve(is_unknown, gmm_probs, pos_label=1)
    auroc_gmm = sk_auc(fpr_gmm, tpr_gmm)
    
    axes[1].plot(fpr_kl, tpr_kl, color='steelblue', lw=2, label=f'KL Value (AUROC={auroc_kl:.4f})')
    axes[1].plot(fpr_gmm, tpr_gmm, color='coral', lw=2, label=f'GMM Prob (AUROC={auroc_gmm:.4f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve (GMM vs Fixed Threshold)')
    axes[1].legend(loc='lower right')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'kl_distribution_auroc_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: kl_distribution_auroc_{timestamp}.png')
    
    # ===== Plot 2: Feature Space Analysis =====
    metrics = compute_feature_metrics(features, labels, args.shared_classes)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    classes = list(range(args.shared_classes)) + [args.shared_classes]
    intra_vals = [metrics['intra_per_class'].get(c, 0) for c in classes]
    colors = ['steelblue'] * args.shared_classes + ['coral']
    labels_bar = [f'C{i}' for i in range(args.shared_classes)] + ['UNK']
    axes[0].bar(range(len(classes)), intra_vals, color=colors, alpha=0.8)
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_xticklabels(labels_bar)
    axes[0].set_ylabel('Intra-class Distance')
    axes[0].set_title('Intra-class Distance per Class')
    axes[0].grid(axis='y', alpha=0.3)
    
    n_classes = args.shared_classes + 1
    inter_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            key1 = f'{i}-{j}'
            key2 = f'{j}-{i}'
            if key1 in metrics['inter_per_pair']:
                inter_matrix[i, j] = metrics['inter_per_pair'][key1]
            elif key2 in metrics['inter_per_pair']:
                inter_matrix[i, j] = metrics['inter_per_pair'][key2]
    
    im = axes[1].imshow(inter_matrix, cmap='YlOrRd', aspect='auto')
    axes[1].set_xticks(range(n_classes))
    axes[1].set_yticks(range(n_classes))
    axes[1].set_xticklabels([f'C{i}' for i in range(args.shared_classes)] + ['UNK'])
    axes[1].set_yticklabels([f'C{i}' for i in range(args.shared_classes)] + ['UNK'])
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title('Inter-class Distance Heatmap')
    
    summary_vals = [metrics['known_intra'], metrics['unknown_intra'],
                    metrics['known_known_inter'], metrics['known_unknown_inter']]
    summary_names = ['Known\nIntra', 'Unknown\nIntra', 'Known-Known\nInter', 'Known-Unknown\nInter']
    axes[2].bar(range(4), summary_vals, color=['steelblue', 'coral', 'green', 'red'], alpha=0.8)
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(summary_names)
    axes[2].set_ylabel('Distance')
    axes[2].set_title(f'Summary (Silhouette={metrics["silhouette"]:.3f})')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'feature_space_analysis_{timestamp}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: feature_space_analysis_{timestamp}.png')
    
    # ===== Print metrics table =====
    print('\n' + '='*70)
    print('Quantitative Metrics')
    print('='*70)
    
    print('\n--- GMM vs Fixed Threshold ---')
    print(f'KL Value AUROC: {auroc_kl:.4f}')
    print(f'GMM Probability AUROC: {auroc_gmm:.4f}')
    print(f'KL Known μ: {kl_known.mean():.4f}, σ: {kl_known.std():.4f}')
    print(f'KL Unknown μ: {kl_unknown.mean():.4f}, σ: {kl_unknown.std():.4f}')
    print(f'Fixed Threshold (q={args.fixed_quantile}): {fixed_threshold:.4f}')
    
    print('\n--- Feature Space Metrics ---')
    print(f'Silhouette Score: {metrics["silhouette"]:.4f}')
    print(f'Known Class Avg Intra-distance: {metrics["known_intra"]:.4f}')
    print(f'Unknown Class Intra-distance: {metrics["unknown_intra"]:.4f}')
    print(f'Known-Known Inter-distance: {metrics["known_known_inter"]:.4f}')
    print(f'Known-Unknown Inter-distance: {metrics["known_unknown_inter"]:.4f}')
    print(f'Separation Ratio (inter_known_unk / known_intra): {metrics["separation_ratio"]:.4f}')
    
    # ===== Save results to JSON =====
    results = {
        'timestamp': timestamp,
        'dataset': 'CIC-IDS2017',
        'parameters': {
            'shared_classes': args.shared_classes,
            'all_classes': args.all_classes,
            'fixed_quantile': args.fixed_quantile,
        },
        'best_epoch': result['best_epoch'],
        'best_hos': float(result['best_hos']),
        'kl_analysis': {
            'kl_known_mean': float(kl_known.mean()),
            'kl_known_std': float(kl_known.std()),
            'kl_unknown_mean': float(kl_unknown.mean()),
            'kl_unknown_std': float(kl_unknown.std()),
            'kl_auroc': float(auroc_kl),
            'gmm_auroc': float(auroc_gmm),
            'fixed_threshold': float(fixed_threshold),
            'gmm_means': gmm.means_.flatten().tolist(),
        },
        'feature_space': metrics,
    }
    
    with open(os.path.join(DATA_DIR, f'results_{timestamp}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nResults saved: {DATA_DIR}/results_{timestamp}.json')
    print(f'All figures saved in: {FIG_DIR}/')

if __name__ == '__main__':
    result = run_single_experiment(total_epochs=args.total_epochs)
    analyze_and_plot(result)
    print('\n' + '='*70)
    print('Done! All outputs saved in:', BASE_DIR)
    print('='*70)
    sys.stdout.close()
    sys.stdout = orig_stdout