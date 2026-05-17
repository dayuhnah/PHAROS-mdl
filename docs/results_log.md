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