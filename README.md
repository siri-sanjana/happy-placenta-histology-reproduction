# HAPPY paper reproduction workspace

This project records reproduction work for Vanea et al., Nature Communications (2024). The workspace keeps an unmodified clone of current HAPPY locally and includes the paper-era v1.0.0 source snapshot needed for this reproduction. The GitHub source snapshot excludes the duplicate current clone, downloaded data, checkpoints and generated bundles.

## Status

The paper, full text, supplementary files and source data are downloaded. Both Zenodo archives are verified and extracted to `data/prepared/`, including the original nuclei/cell annotations, graph embeddings, and graph splits. BioImage Archive slide 1 is size-verified and slide 2 is still downloading; neither is needed for the released embedding-based training. Nuclei initialization smoke training completed. The cell plotting failure was fixed using Matplotlib's Agg backend and optional plot exception handling; a fresh one-epoch cell smoke completed, wrote stats/checkpoints, and generated all four validation confusion matrices. No full paper-schedule training or paper-test evaluation completed: the measured cell smoke epoch took 1,694.55 seconds on CPU, projecting to at least 75 hours for 160 epochs, while only 8.6 GiB remained free. The graph stage is blocked by unavailable PyG operators on this Mac.

The paper-era source is required for RetinaNet/ResNet-50/ClusterGCN. Current HAPPY main is v3 code and is not compatible with the 2024 paper protocol.

## Colab artifacts

The notebook and generated bundle in the local workspace were not used for the Apple M3 run and are intentionally excluded from this source repository snapshot. Reproduction here uses the locally prepared public patch data and the paper-era v1.0.0 implementation.

## Layout

- `references/`: protocol record and publication links; downloaded paper files are local-only.
- `original_repository/`: unmodified clone of official current `main` (local-only; not included in the source snapshot).
- `paper_compatible_repository/`: extracted `v1.0.0` paper-era snapshot; commit hash is in `COMMIT.txt`.
- `data/raw_public/`: original public download archives and BioImage Archive slides (local-only).
- `data/`: extracted/processed data, kept out of source directories and excluded from Git.
- `environment/`: pinned dependency and hardware notes.
- `reports/`: download state, exclusions, and paper comparison.
- `results/`, `logs/`: training outputs when runs happen; smoke logs are included, model weights and generated result files are excluded.

## Setup

Use Python 3.10 in an isolated environment. The paper used PyTorch 2.0.1/torchvision 0.15.2/PyG 2.3.1. On the current Apple M3 host, CUDA and MPS are unavailable; full training is expected to be substantially slower than the paper's A100 runs.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
python -m pip install -e paper_compatible_repository
bash scripts/download_pretrained_weights.sh
# Required by graph sampling; extensions must match torch 2.0.1.
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

Native WSI readers (libvips, OpenSlide, Bio-Formats/Java) are only needed to rebuild upstream image features from whole slides. The released training tiles and graph embeddings are sufficient for the three released-data model stages.

Raw WSI readers are optional: install `environment/requirements-wsi.txt` only after setting up the native reader libraries and a compatible JDK/JVM. On this host the Bio-Formats dependency currently fails to build because it cannot find a compatible JVM; core training does not require it.

PyG's paper-era optional graph extensions do not publish matching macOS arm64 wheels on this host; pip may attempt a source build. The graph stage needs these extensions. If installation fails, retain the exact error in `environment/compatibility_notes.md` and run the graph stage on compatible Linux/A100 hardware rather than silently upgrading PyTorch/PyG.

The official Makefile extension command stopped at `pyg_lib` because no arm64 wheel exists. The pinned source-build attempt is recorded in `environment/compatibility_notes.md` and failed against this host's Xcode libc++.

For a Linux x86_64 CUDA 11.8 host, create a fresh Python 3.10 environment and install the paper-era stack with exact versions (the PyG wheel index uses its compatible 2.0.0 ABI build for the paper's torch 2.0.1):

```sh
conda create -n happy-paper python=3.10.13 -y
conda activate happy-paper
python -m pip install --upgrade pip
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-geometric==2.3.1
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
python -m pip install -r environment/requirements.txt
python -m pip install -e paper_compatible_repository
python -c 'import torch, torch_cluster, torch_sparse; from torch_geometric.data import Data; from torch_geometric.nn import SAGEConv; from torch_geometric.loader import ClusterData; print(torch.__version__, torch.cuda.is_available()); print(torch_cluster.knn_graph(torch.randn(8, 2), 2).shape); x=torch.randn(8, 4); edge=torch.tensor([[0,1,2,3,4,5,6,7],[1,2,3,4,5,6,7,0]]); print(SAGEConv(4, 8)(x, edge).shape)'
```

The graph smoke command must complete and report CUDA available before attempting the 2,000-epoch graph run. Do not install PyG's current/latest packages into the paper-era environment.

The downloader stores the official ImageNet ResNet-101/50 initialisation weights in writable workspace-local paths and checks the published SHA256 prefixes. The runner uses a temporary symlink for the paper-era RetinaNet loader and a workspace-local `TORCH_HOME` for ResNet-50.

## Data

Zenodo record 10535021 contains the training tiles/annotations, graph embeddings, graph split regions, a demo slide and released weights in two zip files. Both archives are verified and extracted to `data/prepared/`; see `reports/download_manifest.md`. To verify the archived source files again:

```sh
stat -f '%N %z bytes' data/raw_public/HAPPYData-20240119T100520Z-*.zip
md5 data/raw_public/HAPPYData-20240119T100520Z-001.zip
md5 data/raw_public/HAPPYData-20240119T100520Z-002.zip
unzip -t data/raw_public/HAPPYData-20240119T100520Z-001.zip
unzip -t data/raw_public/HAPPYData-20240119T100520Z-002.zip
.venv/bin/python scripts/prepare_public_data.py
```

If a transfer stops, resume the same official URLs with `bash scripts/download_public_data.sh`; it uses curl range-resume and skips complete files. A transfer session is currently active; check its status before starting another. The preparation script independently checks the expected byte counts, both published MD5 values, archive integrity, and member paths before extracting/merging into `data/prepared/`. It refuses to overwrite an existing extraction.

The two WSI files under `data/raw_public/bioimage_archive/` are from S-BIAD1045 (CC BY 4.0). They are not needed for graph training from the released embeddings.

## Experiments

The paper schedules are described in `references/reproduction_reference.md`: RetinaNet 40 + 60 epochs; ResNet-50 ImageNet-initialised 60 + 100 epochs; randomly initialised ClusterGCN 2,000 epochs. The supplied runner follows v1.0.0's implementation; the paper's COCO wording for RetinaNet conflicts with that code's ImageNet backbone initialization. Use the original paper-era scripts and exact released splits once the archives are verified. The Zenodo trained weights are for inference/debug comparisons only and must never be reported as newly trained checkpoints.

The original scripts are in `paper_compatible_repository/`:

- `nuc_train.py` and `analysis/evaluation/evaluate_nuclei_model.py`
- `cell_train.py` and `analysis/evaluation/evaluate_cell_model.py`
- `graph_train.py` and `analysis/evaluation/evaluate_graph_model.py`

The supplied paper-era README documents the original file layout and command options. Do not create random replacement test splits. The paper's graph test node count differs across its abstract/results/README and Supplementary Table 7; report the count produced by the original split and evaluator and explain the discrepancy.

After data verification and extraction, run a smoke test before each full stage, then the exact stage command:

```sh
.venv/bin/python scripts/run_public_reproduction.py --stage nuclei --smoke-test
.venv/bin/python scripts/run_public_reproduction.py --stage nuclei
.venv/bin/python scripts/run_public_reproduction.py --stage cell --smoke-test
.venv/bin/python scripts/run_public_reproduction.py --stage cell
.venv/bin/python scripts/run_public_reproduction.py --stage graph --smoke-test
.venv/bin/python scripts/run_public_reproduction.py --stage graph
```

Each full-stage command preserves the paper epoch counts. To run the public pipeline sequentially, use `.venv/bin/python scripts/run_public_reproduction.py`; if a stage fails, the runner records it and still attempts the other stages. If the archive uses different graph annotation/split filenames, pass `--graph-annotations`, `--val-split`, and `--test-split` with the exact names after inspecting the archive.

On this Apple Silicon host, `--stage all` attempts nuclei and cell first, then records the graph stage as blocked because paper-era `torch_cluster` and `torch_sparse` cannot be installed with the present PyG/Xcode combination. CPU training remains far slower than the paper's A100 schedule. Run the graph stage on a compatible Linux x86_64/CUDA environment after its operator smoke check passes.

The local cell plotting fix sets `MPLBACKEND=Agg` for runner-launched child processes and makes validation confusion plots nonfatal. Paper-era test evaluators now save cell prediction arrays/logits and numerical metrics before optional plots; nuclei evaluation saves per-image point predictions and metric JSON. A post-fix one-epoch run is recorded in `logs/repro-cell-plotfix-smoke.log`. It completed in 1,694.55 seconds on CPU. The validation-output path was exercised separately against the smoke checkpoint and is recorded in `logs/repro-cell-validation-output-smoke.log`; these are smoke validation outputs only. The 60+100 epoch cell schedule was not started because this measured throughput implies at least about 75 hours, before additional variation in the unfrozen phase, and checkpoint accumulation could exceed available disk. No smoke metric is entered as a reproduced paper result. No Colab notebook or bundle was used for the local run.

The fresh smoke command, run from `paper_compatible_repository/`, was:

```sh
MPLBACKEND=Agg ../.venv/bin/python cell_train.py \
  --project-name placenta --organ-name placenta \
  --annot-dir annotations/cell_class \
  --dataset-names hmc --dataset-names uot --dataset-names nuh \
  --batch 600 --val-batch 600 --learning-rate 0.0001 \
  --decay-gamma 0.5 --step-size 20 --num-workers 0 \
  --exp-name repro-cell-plotfix-smoke --epochs 1 --init-from-inc --frozen
```

Smoke mode runs one epoch of the initialization phase for nuclei and cells, or one epoch from scratch for graph, on the released training split. Smoke checkpoints are labelled `-smoke` and are not paper results. It does not exercise the higher-rate nuclei fine-tuning phase, since a one-epoch warm-up is too short to make that phase's loss meaningful. The runner saves console logs under `logs/`, and writes loss plots from original training-stat CSVs under `results/loss_curves/`. Inspect the log and latest checkpoint after interruption. Upstream v1.0.0 does not save optimizer/scheduler/RNG state, so exact full-run resumption is unavailable; the runner refuses to restart an existing output directory to prevent an accidental duplicate run. A continuation from model weights only would be a modified training trajectory and must be reported as such.

## Reproduction reports

- `reports/paper_vs_reproduction.csv` intentionally has blank reproduced values until actual predictions are evaluated.
- `reports/excluded_experiments.md` records cohort analyses requiring unreleased clinical WSIs/metadata or human labels.
- `reports/download_manifest.md` tracks official source, expected checksums and retrieval status.

After entering only measured metrics in the CSV, render the Markdown comparison with `.venv/bin/python scripts/render_comparison_report.py`. The renderer never supplies target scores for reproduced values.

Exact reproduction must not be claimed until downloads, archive integrity, training, test predictions and metric comparisons are complete.
