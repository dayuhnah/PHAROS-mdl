# PHAROS Results Log

## Experiment 1: Baseline Ablation on DepMap PRISM

Dataset:
- DepMap PRISM primary screen
- 10,000 sampled response pairs
- 559 cell lines with expression
- 19,205 expression genes
- 29 curated resistance genes
- Target: log fold-change

Split:
- Random 80/20 train-validation split

Models:
1. Drug Only
2. Drug + Expression
3. PHAROS Baseline: Drug + Expression + Resistance

Final epoch results:

| Model | Val Loss | RMSE | MAE | R2 | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Drug Only | 0.4987 | 0.7062 | 0.4819 | 0.3389 | 0.6154 | 0.2911 |
| Drug + Expression | 0.4771 | 0.6908 | 0.4694 | 0.3674 | 0.6113 | 0.2713 |
| PHAROS Baseline | 0.4874 | 0.6982 | 0.4762 | 0.3538 | 0.6261 | 0.2773 |

Best observed PHAROS epoch:
- Epoch 4
- Val Loss: 0.4644
- RMSE: 0.6815
- R2: 0.3843
- Pearson: 0.6252

Observation:
PHAROS achieved the best Pearson correlation at the final epoch and the best overall validation loss/RMSE/R2 at epoch 4. This suggests resistance-aware features may improve predictive alignment, but early stopping is needed to prevent overfitting.

## Experiment 2: Ablation with Early Stopping

Dataset:
- DepMap PRISM primary screen
- 10,000 sampled response pairs
- 559 cell lines with expression
- 19,205 expression genes
- 29 curated resistance genes
- Target: log fold-change

Split:
- Random 80/20 train-validation split

Early stopping:
- Max epochs: 10
- Patience: 3
- Best model selected by validation loss

| Model | Best Epoch | Train Loss | Val Loss | RMSE | MAE | R2 | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Drug Only | 7 | 0.2850 | 0.4695 | 0.6852 | 0.4671 | 0.3776 | 0.6316 | 0.3035 |
| Drug + Expression | 6 | 0.3052 | 0.4593 | 0.6778 | 0.4592 | 0.3910 | 0.6293 | 0.2813 |
| PHAROS Baseline | 4 | 0.3559 | 0.4507 | 0.6713 | 0.4603 | 0.4025 | 0.6367 | 0.2609 |

Observation:
PHAROS achieved the best validation loss, RMSE, R², and Pearson correlation, suggesting that the resistance-aware branch adds useful predictive signal beyond drug fingerprints and full gene-expression features. However, Drug Only achieved the highest Spearman correlation, indicating that future work should improve rank-order prediction of response strength.

## Experiment 3: Unseen-Drug Split

Dataset:
- DepMap PRISM primary screen
- 10,000 sampled response pairs
- Split by unique `broad_id`
- Validation drugs are completely unseen during training
- Target: log fold-change

| Model | Best Epoch | Train Loss | Val Loss | RMSE | MAE | R2 | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Drug Only | 1 | 0.6309 | 0.6719 | 0.8197 | 0.5377 | 0.1001 | 0.3328 | 0.1292 |
| Drug + Expression | 2 | 0.5144 | 0.6752 | 0.8217 | 0.5312 | 0.0957 | 0.3429 | 0.1141 |
| PHAROS: Drug + Expression + Resistance | 2 | 0.5165 | 0.6814 | 0.8255 | 0.5347 | 0.0875 | 0.3421 | 0.1063 |

Observation:
Performance dropped substantially compared with the random split, confirming that unseen-drug prediction is a much harder generalization setting. PHAROS performed similarly to the Drug + Expression baseline but did not improve unseen-drug performance. This suggests that the current Morgan fingerprint representation is insufficient for strong new-compound generalization, motivating a graph-based molecular encoder such as GATv2.