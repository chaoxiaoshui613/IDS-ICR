# IDS-ICR: Intrusion Detection with Inter-Class Relationships

Official implementation of the paper *"Intrusion Detection System for Open Network Scenarios: A Known/Unknown Attack Detection Method Based on Inter-Class Relationships."*

## Overview

This repository provides the training and evaluation code for an open-set intrusion detection method that simultaneously classifies known attacks and detects unknown attacks. The method introduces a virtual class into the softmax layer to reserve feature space for unknowns, and uses soft-label prototypes to encode inter-class relationships. KL divergence between test predictions and these prototypes is fitted by a Gaussian Mixture Model to adaptively separate known from unknown samples.

## Files

| File | Description |
|------|-------------|
| `main.py` | Complete implementation: model classes, loss functions, training loop, and evaluation |
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


