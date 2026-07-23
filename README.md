[English](#english) | [中文](#chinese)

---

<a id="english"></a>

# English

# IDS-ICR: Intrusion Detection with Inter-Class Relationships

Official implementation of the paper *"Intrusion Detection System for Open Network Scenarios: A Known/Unknown Attack Detection Method Based on Inter-Class Relationships."*

## Overview

This repository provides the training and evaluation code for an open-set intrusion detection method that simultaneously classifies known attacks and detects unknown attacks. The method introduces a virtual class into the softmax layer to reserve feature space for unknowns, and uses soft-label prototypes to encode inter-class relationships. KL divergence between test predictions and these prototypes is fitted by a Gaussian Mixture Model to adaptively separate known from unknown samples.

## Files

| File | Description |
|------|-------------|
| `main.py` | Complete implementation: model classes, loss functions, training loop, and evaluation |
| `ratio_experiment.py` | Ratio ablation experiment for anomaly detection pre-filtering duration |
| `ratio__nopre.py` | Ablation variant that removes anomaly detection pre-filtering, using only GMM for known/unknown separation |
| `experiment_kl_inter.py` | KL divergence distribution visualization with intra-class and inter-class distance analysis |
| `requirements.txt` | Python dependencies |

## Code Structure

`main.py` is organized into the following sections (in order of appearance):

| Section | Lines (approx.) | Description |
|---------|-----------------|-------------|
| Imports & GPU setup | 1–30 | Library imports and automatic CUDA/CPU detection |
| Argument parsing | 35–55 | Command-line arguments via `argparse` |
| Logging | 70–115 | `Tee` class for simultaneous console and file output |
| Data utilities | 117–210 | `one_hot`, label transforms, `get_split_dataset_info`, `CustomDataset` |
| Centroids | 236–310 | Source and target class centroid tracking with dynamic update |
| Feature extractor | 418–466 | `TabularAutoencoder`: encoder-decoder structure mapping input to 256-dim features |
| Classifier | 469–523 | `CLS`: softmax classifier with `virt_forward` for virtual class logit injection |
| Adversarial network | 527–575 | `LargeAdversarialNetwork` with gradient reversal for domain adaptation |
| Anomaly detector | 577–614 | `AnomalyDetector` and `BinaryClassifier` for benign/attack separation |
| Utility classes | 618–915 | `Accumulator`, `OptimWithSheduler`, `DomainBus`, `LossCounter`, `Logger` |
| Loss functions | 918–1028 | `CrossEntropyLoss`, `BCELossForMultiClassification`, `EntropyLoss` |
| Evaluation helpers | 1030–1195 | `extended_confusion_matrix`, `cal_sim`, `calculate_open_set_metrics` |
| Initial pre-filtering | 1395–1530 | Anomaly detection model training, test set clustering, bipartite graph matching for virtual class initialization |
| Optimizer setup | 1570–1590 | SGD with inverse decay scheduling for discriminator, feature extractor, and classifier |
| Training loop | 1595–1815 | Per-epoch: KL divergence computation, GMM fitting, loss calculation, centroid update, evaluation |
| Cleanup | 1832–1836 | Log file closing, stdout restoration |

## Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA (optional; CPU mode auto-detected)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python main.py \
    --source train_0_6.csv \
    --target test_0_8.csv \
    --shared_classes 7 \
    --all_classes 8 \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --total_epochs 100 \
    --data_dir /path/to/data/ \
    --log_dir /path/to/logs/ \
    --name experiment_name
```

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--source` | Training CSV (known classes only) | `source.csv` |
| `--target` | Test CSV (known + unknown) | `target.csv` |
| `--shared_classes` | Number of known classes including benign | 7 |
| `--all_classes` | Total classes (known + unknown) | 8 |
| `--batch_size` | Batch size | 64 |
| `--learning_rate` | Learning rate | 5e-5 |
| `--total_epochs` | Number of training epochs | 100 |
| `--data_dir` | Directory containing CSV files | — |
| `--log_dir` | Directory for log output | — |
| `--name` | Experiment name | `1` |

## Data Format

Each CSV file should have features in all columns except the last, and integer class labels in the last column. Training data contains only known classes (labels 0 to shared_classes-1). Test data contains both known classes and unknown attacks (labels shared_classes and above).

## Output

Training logs are saved to `--log_dir` with timestamps. Each epoch prints:

- **OS**: Overall accuracy
- **OS\***: Average accuracy across known classes
- **UNK**: Unknown attack detection accuracy
- **HOS**: Harmonic mean of OS* and UNK
- Per-loss values (ce, entropy, virtual, ce_ep, adv)
- Normalized confusion matrix

The best model checkpoint (by HOS) is automatically tracked.

## Metrics

- **Accuracy**: Per-class averaged or weighted classification accuracy
- **UNK**: Accuracy on unknown attack samples (diagonal of unknown class row in confusion matrix)
- **OS\***: Mean per-class accuracy across known classes
- **HOS** = 2 × OS* × UNK / (OS* + UNK)
- **F1**: Macro-averaged F1 score



---

<a id="chinese"></a>

# 中文

# IDS-ICR：基于类间关系的入侵检测系统

论文 *"面向开放网络场景的入侵检测系统：基于类间关系的已知/未知攻击检测方法"* 的官方实现。

## 概述

本仓库提供了一种开放集入侵检测方法的训练与评估代码，该方法可同时对已知攻击进行细粒度分类并检测未知攻击。方法通过在 softmax 层中引入虚拟类别为未知攻击预留特征空间，并利用软标签原型编码已知类别的类间关系。测试样本预测与原型之间的 KL 散度通过高斯混合模型进行自适应拟合，从而区分已知与未知样本。

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | 完整实现：模型类、损失函数、训练循环与评估 |
| `ratio_experiment.py` | 异常检测预过滤时长的比例消融实验 |
| `ratio__nopre.py` | 消融变体：移除异常检测预过滤，仅使用 GMM 区分已知/未知 |
| `experiment_kl_inter.py` | KL 散度分布可视化及类内、类间距离分析 |
| `requirements.txt` | Python 依赖项 |

## 代码结构

`main.py` 按以下顺序组织：

| 模块 | 大致行号 | 说明 |
|------|----------|------|
| 导入与 GPU 配置 | 1–30 | 库导入与 CUDA/CPU 自动检测 |
| 参数解析 | 35–55 | 通过 `argparse` 解析命令行参数 |
| 日志 | 70–115 | `Tee` 类实现控制台与文件同步输出 |
| 数据工具 | 117–210 | `one_hot`、标签变换、`get_split_dataset_info`、`CustomDataset` |
| 质心 | 236–310 | 源域与目标域类别质心跟踪与动态更新 |
| 特征提取器 | 418–466 | `TabularAutoencoder`：编码器-解码器结构，将输入映射至 256 维特征 |
| 分类器 | 469–523 | `CLS`：softmax 分类器，含 `virt_forward` 实现虚拟类 logit 注入 |
| 对抗网络 | 527–575 | `LargeAdversarialNetwork`，含梯度反转层用于域适应 |
| 异常检测器 | 577–614 | `AnomalyDetector` 与 `BinaryClassifier`，用于良性/攻击流量分离 |
| 工具类 | 618–915 | `Accumulator`、`OptimWithSheduler`、`DomainBus`、`LossCounter`、`Logger` |
| 损失函数 | 918–1028 | `CrossEntropyLoss`、`BCELossForMultiClassification`、`EntropyLoss` |
| 评估辅助 | 1030–1195 | `extended_confusion_matrix`、`cal_sim`、`calculate_open_set_metrics` |
| 初始预过滤 | 1395–1530 | 异常检测模型训练、测试集聚类、二分图匹配初始化虚拟类 |
| 优化器配置 | 1570–1590 | 判别器、特征提取器、分类器的 SGD 逆衰减调度 |
| 训练循环 | 1595–1815 | 每轮：KL 散度计算、GMM 拟合、损失计算、质心更新、评估 |
| 清理 | 1832–1836 | 日志文件关闭、stdout 恢复 |

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA（可选，自动检测 CPU 模式）

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
python main.py \
    --source train_0_6.csv \
    --target test_0_8.csv \
    --shared_classes 7 \
    --all_classes 8 \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --total_epochs 100 \
    --data_dir /path/to/data/ \
    --log_dir /path/to/logs/ \
    --name experiment_name
```

## 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source` | 训练集 CSV（仅已知类） | `source.csv` |
| `--target` | 测试集 CSV（已知 + 未知） | `target.csv` |
| `--shared_classes` | 已知类别数（含良性） | 7 |
| `--all_classes` | 总类别数（已知 + 未知） | 8 |
| `--batch_size` | 批次大小 | 64 |
| `--learning_rate` | 学习率 | 5e-5 |
| `--total_epochs` | 训练轮数 | 100 |
| `--data_dir` | CSV 文件所在目录 | — |
| `--log_dir` | 日志输出目录 | — |
| `--name` | 实验名称 | `1` |

## 数据格式

每个 CSV 文件的最后一列为整数类别标签，其余列为特征。训练数据仅包含已知类别（标签 0 至 shared_classes-1）。测试数据同时包含已知类别和未知攻击（标签 shared_classes 及以上）。

## 输出

训练日志按时间戳保存至 `--log_dir`。每轮输出：

- **OS**：总体准确率
- **OS\***：已知类平均准确率
- **UNK**：未知攻击检测准确率
- **HOS**：OS* 与 UNK 的调和平均
- 各项损失值（ce, entropy, virtual, ce_ep, adv）
- 归一化混淆矩阵

最优模型（按 HOS）自动保存。

## 评估指标

- **Accuracy**：逐类平均或加权分类准确率
- **UNK**：未知攻击样本的检测准确率
- **OS\***：已知类别的平均准确率
- **HOS** = 2 × OS* × UNK / (OS* + UNK)
- **F1**：宏平均 F1 分数

