# Reproduction reference

## Paper and source

- **Title:** Mapping cell-to-tissue graphs across human placenta histology whole slide images using deep learning with HAPPY
- **Authors:** Claudia Vanea, Jelisaveta Džigurski, Valentina Rukins, Omri Dodi, Siim Siigur, Liis Salumäe, Karen Meir, W. Tony Parks, Drorith Hochner-Celnikier, Abigail Fraser, Hagit Hochner, Triin Laisk, Linda M. Ernst, Cecilia M. Lindgren, and Christoffer Nellåker.
- **Journal/year:** Nature Communications 15, 2710 (2024)
- **DOI:** [10.1038/s41467-024-46986-2](https://doi.org/10.1038/s41467-024-46986-2)
- **Paper:** <https://pmc.ncbi.nlm.nih.gov/articles/PMC10978962/>
- **Repository:** <https://github.com/Nellaker-group/happy>
- **Repository paper-era tag:** `v1.0.0`, commit `9ece11c1988cf498dc2e99484a990394b3139efd` (2024-01-14). Snapshot preserved at `paper_compatible_repository/`.
- **Current clone:** `original_repository/`, cloned from `main` at `851613d2aab43ac888f66aca93f7749de28fd62c` (2026-07-01). This current v3 code is not paper-compatible: it uses YOLO26 for nuclei and has a revised tissue graph implementation. It remains unmodified.
- **Code DOI:** <https://doi.org/10.5281/zenodo.10529239>

## Public data

- Training/validation/test tiles, annotations, graph embeddings, graph split regions, a demo slide, and released weights: [Zenodo record 10535021](https://zenodo.org/records/10535021), version record `10535022`.
- Two source WSIs for graph training: [BioImage Archive S-BIAD1045](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1045), CC BY 4.0. Archive file manifest reports `96-2019-09-05 22.47.40.ndpi` (2,686,927,912 bytes) and `59B.svs` (2,929,818,169 bytes).
- Zenodo release license: CC BY 4.0. Archives and published MD5 checksums:
  - `HAPPYData-20240119T100520Z-001.zip`, 2,141,793,908 bytes, `e8af7d9a349051e209c9d982bc44edcf`
  - `HAPPYData-20240119T100520Z-002.zip`, 452,945,061 bytes, `e39adcab4ea865f8d498d62ced6b1ca6`
- Zenodo describes the archives as containing `annotations`, `datasets`, `embeddings`, `graph_splits`, `slides`, and `trained_models`; same-named directories from the two archives must be combined. The graph model can be trained from the released embeddings and split files. Using those inputs does **not** reproduce raw-slide cell detection/classification for the two graph slides. The separate BioImage Archive WSIs enable that upstream reconstruction but are not needed to train the graph stage from the released embeddings.
- Path compatibility note: the v1.0.0 README places graph HDF5 files at `projects/placenta/embeddings/`, but the paper-era `happy/utils/hdf5.py` resolves them under `projects/placenta/results/embeddings/` using paths from `happy/db/main.db`. The workspace runner creates a symlink at the code-required location while keeping extracted files under `data/prepared/`.
- The paper states that other in-house placental WSIs and associated clinical data are unavailable under the governing ethics approvals and data transfer agreements.

## Architecture and paper protocol

### A. Nuclei localisation

- RetinaNet with ResNet-101 backbone; input tiles 1600×1200 px with 200 px overlap; single nuclei class.
- The **paper Methods** say COCO initialisation; Adam, focal loss. Fine-tune 40 epochs at LR 0.0001, LR ×0.5 every 20 epochs; select best validation F1. Then fully train 60 epochs at LR 0.001 with the same decay and select best validation F1.
- **Material paper/code discrepancy:** the v1.0.0 implementation's `init_from_inc` path loads only the torchvision ResNet-101 ImageNet backbone (`resnet101-5d3b4d8f.pth`) with `strict=False`; the RetinaNet feature pyramid and heads are newly initialised. The paper's data statement also calls the ResNet-101 weights ImageNet weights, while Methods and README call them COCO. The supplied runner follows the preserved v1.0.0 implementation (ImageNet backbone + new detector heads), not an unavailable/full COCO RetinaNet checkpoint. This is not verified exact COCO initialisation.
- Heavy geometric/colour/H&E augmentation (paper Supplementary Table 6). Duplicate predictions within 4 px are removed.
- Point matching: a predicted centroid is correct within 3.3 μm; paper-compatible code evaluator uses 30 px at 0.1109 μm/px (3.327 μm).
- Split in Supplementary Table 7: 11,755/2,374/2,754 nuclei (train/validation/test); image patches 176/38/38.
- Paper states these splits were randomly generated at about 70/15/15 per dataset/model. The publication does not describe a patient-level holdout; do not characterize these test sets as patient-independent validation.
- Reported test F1: **0.884**, 2,754 nuclei in 38 test images (Results, “Evaluation of model performance”, Table 1).
- Release split check after extraction: HMC and UOT nuclei test annotation CSVs contain 890 + 1,864 = 2,754 nucleus rows; 12 empty-image rows are in 12 separate patches, giving 38 test patches total. Cell test CSVs contain 686 HMC + 1,695 UOT + 362 NUH = 2,743 annotated cells, matching the reported test count.
- Cell class IDs in the original `happy/organs.py`: 0 CYT, 1 FIB, 2 HOF, 3 SYN, 4 VEN, 5 MAT, 6 VMY, 7 WBC, 8 MES, 9 EVT, 10 KNT.

### B. Cell classification

- ResNet-50, 200×200 px centred on nuclei, 11 cell types, penultimate 64-dimensional embedding.
- ImageNet initialisation; Adam, cross-entropy. Fine-tune 60 epochs at LR 0.0001, LR ×0.5 every 20 epochs; select best validation accuracy. Then train all layers 100 epochs at LR 0.0001 with the same decay; select best validation accuracy. Minority classes oversampled; same image augmentation family as nuclei localisation.
- Test set in Supplementary Table 7: **2,743 cells** total; class-specific counts are in that table.
- Reported: accuracy **84.29%**, top-2 accuracy **94.90%**, macro ROC-AUC **0.9773** (Results, “Evaluation of model performance”, Table 1). Paper-era evaluation code computes macro one-vs-one ROC-AUC (`multi_class="ovo"`).

### C. Tissue classification

- Build cell graph from 64-dimensional cell embeddings and coordinates. Undirected intersection of kNN (k=5) and Delaunay edges with self-loops; syncytial knot post-processing is part of the repository workflow.
- Randomly initialised inductive ClusterGCN: 16 GraphSAGEConv layers, 256 hidden units, dropout and batch normalisation; Adam, custom weighted cross entropy, LR 0.001, batch size 200, 400-neighbour/subgraph sampling, 2,000 epochs. Select best validation accuracy; validation score uses full neighbourhood and intersection graph k=8 as described by the paper.
- Nine tissue classes. Train/validation/test regions are spatial patches chosen to exceed the 16-hop receptive field; use the released `graph_splits` rather than random node splits.
- Nine reported classes/IDs in the paper-era organ map: 1 Sprout, 3 TVilli (terminal villi), 5 MIVilli (mature intermediate villi), 6 AVilli (anchoring villi), 7 SVilli (stem villi), 8 Chorion, 9 Maternal (basal plate/septum), 11 Fibrin, 12 Avascular. The source organ map also defines `Unlabelled`, `MVilli`, `ImIVilli`, and `Inflam`; verify how released graph annotations encode these. The evaluator's model head is sized from the full organ map, so do not assume it has only nine output indices.
- Reported: accuracy **68.34%**, top-2 **91.14%**, top-3 **97.10%**, macro ROC-AUC **0.8868** (Results, “Evaluation of model performance”, Table 1; repository README repeats these values). Paper-era evaluation code computes macro one-vs-one ROC-AUC (`multi_class="ovo"`).
- **Published count discrepancy to resolve against released graph files/evaluator:** main text and repository README say 149,425 evaluated nodes; abstract says 179,095; Supplementary Table 7 test counts sum to 179,095. Do not silently select one count. Record the actual count produced by the supplied split/evaluator and state which published location it matches.

## Paper-reported environment

Python 3.10.13; PyTorch 2.0.1; torchvision 0.15.2; PyTorch Geometric 2.3.1; libvips 8.9.2; OpenSlide 3.4.1; Bio-Formats 6.11; pyvips 2.1.14; openslide-python 1.2.0; python-bioformats 4.0.7; h5py 3.8.0; albumentations 1.3.0; pandas 2.0.1; numpy 1.24.1; scikit-image 0.22.0; scikit-learn 1.3.1; matplotlib 3.7.1; peewee 3.16.2; typer 0.9.0; visdom 0.2.4; umap-learn 0.5.3; seaborn 0.12.2. Authors report training on one NVIDIA A100.

## Publicly reproducible scope

The released tile datasets, labels, pretrained initialisation paths, graph embeddings, annotation files, graph split definitions, and source model code make the three supervised training/evaluation stages publicly reproducible in principle. Re-running graph training from released embeddings is a reproduction of the graph stage, not raw-slide upstream inference. The released trained weights are retained only for comparison/inference checks and are not reproduction training outputs.

The paper-era code loads torchvision ImageNet ResNet-101 backbone weights for RetinaNet while its Methods/README call the initialization COCO. The runner follows code behavior; this remains a documented protocol ambiguity.

The supplementary GTEx/other-organ generalisation analyses, healthy cohort slide-level quantification, infarction cohort comparisons, and pathologist labelling/agreement experiment are not part of the default reproducible pipeline: required WSI cohorts, clinical metadata, or human annotation data are not included in the public training release. The two BIA slide files are public under CC BY 4.0 and are kept separate from the restricted cohorts.

## Retrieval and execution status

See `reports/download_manifest.md`, `environment/system_info.txt`, `environment/compatibility_notes.md`, and `reports/paper_vs_reproduction.md`. A URL being public does not imply that a large transfer or training run has completed. This workspace is not an exact reproduction until all data checks, full paper-schedule training, test evaluation, and comparisons are completed.
