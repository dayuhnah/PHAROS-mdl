# PHAROS

PHArmacological Response Oracle with biological Signals.

## Project Aim

This project aims to develop a mechanism-aware model for anticancer drug response prediction by integrating:

- drug molecular graph features
- cancer cell-line gene expression
- resistance mechanism gene panels
- explainable AI for molecular and gene-level interpretation

## Initial MVP

The first version predicts drug sensitivity using:

- Drug SMILES converted into molecular fingerprints
- Cell-line baseline gene expression
- Curated resistance-related gene expression panel

## Planned Extensions

- GATv2 molecular graph encoder
- Bioactivity bridge module
- LINCS L1000 gene-expression prediction
- GNNExplainer for atom/bond attribution
- Integrated Gradients for resistance gene attribution
- Two-drug combination mode