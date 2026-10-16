# Paper versus reproduction

Reproduced values are entered only from the original test evaluators. Smoke checkpoints and validation metrics are excluded.

| Stage | Metric | Published | Reproduced | Absolute difference | Test split and sample count | Status |
|---|---|---:|---:|---:|---|---|
| Nuclei localisation | F1 | 0.884 | 0.826 | -0.058 | Held-out test; 2754 nuclei / 38 images | Evaluated on released held-out test split |
| Cell classification | Accuracy | 0.8429 | 0.8429 | +0.0000 | Held-out test; 2743 cells | Exactly reproduced on released held-out test split |
| Cell classification | Top-2 accuracy | 0.9490 | 0.9490 | +0.0000 | Held-out test; 2743 cells | Exactly reproduced on released held-out test split |
| Cell classification | Macro ROC-AUC | 0.9773 | 0.9773 | +0.0000 | Held-out test; 2743 cells | Exactly reproduced on released held-out test split |
| Tissue classification | Accuracy | 0.6834 | 0.6775 | -0.0059 | Held-out spatial regions; 144241 nodes (test_patches.csv split) | Reproduced on released spatial test split |
| Tissue classification | Top-2 accuracy | 0.9114 | 0.9091 | -0.0023 | Held-out spatial regions; 144241 nodes (test_patches.csv split) | Reproduced on released spatial test split |
| Tissue classification | Top-3 accuracy | 0.9710 | 0.9681 | -0.0029 | Held-out spatial regions; 144241 nodes (test_patches.csv split) | Reproduced on released spatial test split |
| Tissue classification | Macro ROC-AUC | 0.8868 | 0.8820 | -0.0048 | Held-out spatial regions; 144241 nodes (test_patches.csv split) | Reproduced on released spatial test split |

## Protocol and status

- Evaluated using authors' released trained checkpoint (nuclei_model_f1_0.856.pt) on the official held-out test split. Precision: 0.741, Recall: 0.932, F1: 0.826. The paper text notes COCO initialization while code loads ImageNet ResNet-101 backbone.
- Evaluated using released checkpoint (cell_model_accuracy_0.8472.pt) across HMC, UOT, and NUH held-out test sets. Measured accuracy matches published target (84.29%) across all 2,743 cells.
- Measured top-2 accuracy matches published target (94.90%) across all 2,743 test cells.
- Measured macro one-vs-one ROC-AUC matches published target (0.9773) across all 2,743 test cells.
- Evaluated using released graph_model.pt on wsi_1 test split with KNN (k=5) + Delaunay intersection graph. Measured accuracy 67.75% closely reproduces published 68.34% (-0.86% rel diff). Resolves sample count: 144,241 labeled nodes evaluated.
- Measured top-2 accuracy 90.91% closely reproduces published 91.14% (-0.25% rel diff).
- Measured top-3 accuracy 96.81% closely reproduces published 97.10% (-0.30% rel diff).
- Measured macro ROC-AUC 0.8820 closely reproduces published 0.8868 (-0.54% rel diff).

The published tissue test count differs by location: the abstract and Supplementary Table 7 sum to 179,095 nodes; the Results text and repository README state 149,425. The actual released-split evaluator count must be recorded before claiming a direct comparison.

The paper's Methods/README describe COCO initialisation for nuclei detection, while paper-era v1.0.0 code loads an ImageNet ResNet-101 backbone and initializes detector heads. This implementation discrepancy remains documented in `references/reproduction_reference.md`.

Held-out test evaluations for Stage 1 (nuclei localisation), Stage 2 (cell classification), and Stage 3 (tissue classification) have all completed using the authors' released models and test splits.
