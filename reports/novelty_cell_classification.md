# Novelty Architectural Extension: Residual Channel Attention & Loss Optimization

## Summary of Results (Held-out Test Split: 2,743 cells across HMC, UOT, NUH)

| Model Configuration                            | Accuracy   | Top-2 Accuracy   |   Macro ROC-AUC | Diff vs Baseline (Acc)   | Description                                                             |
|:-----------------------------------------------|:-----------|:-----------------|----------------:|:-------------------------|:------------------------------------------------------------------------|
| Baseline (Authors Released / Replicated)       | 84.29%     | 94.90%           |          0.9773 | +0.00%                   | Original ResNet-50 head (ReLU + Cross-Entropy)                          |
| Baseline (Re-trained MLP)                      | 83.74%     | 94.28%           |          0.9713 | -0.55%                   | Standard MLP head, ReLU activation, Weighted Cross-Entropy Loss         |
| + Mish Activation                              | 83.89%     | 94.31%           |          0.9722 | -0.40%                   | Replaces ReLU with smooth non-monotonic Mish activation                 |
| + Focal Loss                                   | 83.96%     | 94.50%           |          0.9735 | -0.33%                   | Replaces CE with Focal Loss (gamma=2.0) to focus on hard/minority cells |
| + Channel Attention (SE)                       | 83.74%     | 94.20%           |          0.9702 | -0.55%                   | Adds residual Squeeze-and-Excitation channel attention before MLP       |
| Proposed Full Model (Attention + Mish + Focal) | 84.21%     | 94.42%           |          0.9724 | -0.07%                   | Residual Channel Attention + Mish activation + Multi-class Focal Loss   |
