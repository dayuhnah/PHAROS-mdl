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
Performance dropped substantially compared with the random split, confirming that unseen-drug prediction is a much harder generalization setting. PHAROS performed similarly to the Drug + Expression baseline but did not improve unseen-drug performance. This suggests that the current Morgan fingerprint representation is insufficient for strong new-compound generalization, motivating a graph-based molecular encoder such as GATv2.'

## Experiment 4: Unseen-Cell-Line Split

Dataset:
- DepMap PRISM primary screen
- 10,000 sampled response pairs
- Split by unique `depmap_id`
- Validation cell lines are completely unseen during training
- Target: log fold-change

| Model | Best Epoch | Train Loss | Val Loss | RMSE | MAE | R2 | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Drug Only | 3 | 0.3816 | 0.4547 | 0.6743 | 0.4521 | 0.3891 | 0.6254 | 0.2441 |
| Drug + Expression | 5 | 0.3264 | 0.4568 | 0.6759 | 0.4571 | 0.3862 | 0.6296 | 0.2751 |
| PHAROS: Drug + Expression + Resistance | 4 | 0.3608 | 0.4520 | 0.6723 | 0.4532 | 0.3928 | 0.6292 | 0.2579 |

Observation:
PHAROS achieved the best validation loss, RMSE, and R² in the unseen-cell-line setting, suggesting that resistance-aware features help generalize to unseen cancer cell contexts. Drug + Expression achieved the highest Pearson and Spearman correlation, indicating that future PHAROS improvements should refine the resistance branch for better ranking of response strength.

## Experiment 5: Corrected Resistance Feature Separation

In this experiment, the 29 curated resistance genes were removed from the general expression vector and passed only through the dedicated PHAROS resistance branch. This makes the ablation cleaner because the Drug + Expression baseline no longer has direct access to the resistance panel.

### Corrected Random Split

| Model | Best Epoch | Train Loss | Val Loss | RMSE | MAE | R2 | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Drug Only | 5 | 0.3165 | 0.4515 | 0.6719 | 0.4581 | 0.4015 | 0.6381 | 0.2840 |
| Drug + Expression | 6 | 0.3010 | 0.4523 | 0.6725 | 0.4607 | 0.4003 | 0.6400 | 0.2852 |
| PHAROS Baseline | 5 | 0.3285 | 0.4665 | 0.6830 | 0.4653 | 0.3815 | 0.6347 | 0.2748 |

Observation:
After correcting the feature separation, PHAROS did not outperform the simpler baselines in the random split. This suggests that the current resistance branch, which is fused by simple concatenation, is not yet strong enough to consistently improve random drug-cell response prediction.

### Corrected Unseen-Cell-Line Split

| Model | Best Epoch | Train Loss | Val Loss | RMSE | MAE | R2 | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Drug Only | 6 | 0.2939 | 0.4588 | 0.6773 | 0.4542 | 0.3836 | 0.6260 | 0.2860 |
| Drug + Expression | 4 | 0.3555 | 0.4545 | 0.6742 | 0.4409 | 0.3894 | 0.6283 | 0.2687 |
| PHAROS | 5 | 0.3357 | 0.4512 | 0.6717 | 0.4492 | 0.3938 | 0.6290 | 0.2742 |

Observation:
In the unseen-cell-line split, PHAROS achieved the best validation loss, RMSE, R², and Pearson correlation. This suggests that the dedicated resistance branch is more useful for generalizing to unseen cancer cell contexts, which aligns with the biological motivation of the project.