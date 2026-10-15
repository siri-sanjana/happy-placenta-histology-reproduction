# Paper versus reproduction

Reproduced values are entered only from the original test evaluators. Smoke checkpoints and validation metrics are excluded.

| Stage | Metric | Published | Reproduced | Absolute difference | Test split and sample count | Status |
|---|---|---:|---:|---:|---|---|
| Nuclei localisation | F1 | 0.884 | — | — | Held-out test; 2754 nuclei / 38 images | Smoke only; full schedule not run; test not evaluated |
| Cell classification | Accuracy | 0.8429 | — | — | Held-out test; 2743 cells | Smoke only; full schedule not run; test not evaluated |
| Cell classification | Top-2 accuracy | 0.9490 | — | — | Held-out test; 2743 cells | Smoke only; full schedule not run; test not evaluated |
| Cell classification | Macro ROC-AUC | 0.9773 | — | — | Held-out test; 2743 cells | Smoke only; full schedule not run; test not evaluated |
| Tissue classification | Accuracy | 0.6834 | — | — | Held-out spatial regions; Published count discrepancy: 149425 (Results/README) vs 179095 (abstract/Supplementary Table 7); actual evaluator count pending | Not executed |
| Tissue classification | Top-2 accuracy | 0.9114 | — | — | Held-out spatial regions; Published count discrepancy: 149425 (Results/README) vs 179095 (abstract/Supplementary Table 7); actual evaluator count pending | Not executed |
| Tissue classification | Top-3 accuracy | 0.9710 | — | — | Held-out spatial regions; Published count discrepancy: 149425 (Results/README) vs 179095 (abstract/Supplementary Table 7); actual evaluator count pending | Not executed |
| Tissue classification | Macro ROC-AUC | 0.8868 | — | — | Held-out spatial regions; Published count discrepancy: 149425 (Results/README) vs 179095 (abstract/Supplementary Table 7); actual evaluator count pending | Not executed |

## Protocol and status

- Existing one-epoch initialization smoke checkpoint/log; no paper test result. Full 40+60 schedule not started because MPS is unavailable and current free disk is insufficient for potentially retained best checkpoints. Smoke wall time was not reliably recorded.
- Post-fix one-epoch initialization smoke completed on CPU in 1,694.55 s; validation accuracy 0.209 is smoke-only and excluded from reproduced values. Validation outputs/logits and metrics were persisted by a separate evaluator check. Full 60+100 schedule not started: measured rate projects to at least 75.3 h and phase two may be slower; 8.6 GiB free is below potential accumulated checkpoint storage.
- Zenodo graph embeddings/splits are present, but torch_cluster/torch_sparse cannot run on this host; no Linux GPU runtime is available

The published tissue test count differs by location: the abstract and Supplementary Table 7 sum to 179,095 nodes; the Results text and repository README state 149,425. The actual released-split evaluator count must be recorded before claiming a direct comparison.

The paper's Methods/README describe COCO initialisation for nuclei detection, while paper-era v1.0.0 code loads an ImageNet ResNet-101 backbone and initializes detector heads. This implementation discrepancy remains documented in `references/reproduction_reference.md`.

No exact reproduction is claimed: neither full paper schedule nor either original held-out test evaluation ran. The 59B.svs transfer remains irrelevant to these two released-patch experiments.
