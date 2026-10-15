# HAPPY_LOCAL_OUTPUT_PATCH
#!/usr/bin/env python3
"""Run one or all paper-era HAPPY experiments using the official released splits."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "paper_compatible_repository"
PROJECT = REPO / "projects" / "placenta"
PYTHON = ROOT / ".venv" / "bin" / "python"
SMOKE = False
SUFFIX = ""
LOADER_WORKERS = 0 if platform.system() == "Darwin" else None


def need(path: Path, what: str) -> None:
    if not path.exists():
        raise SystemExit(f"Required {what} is missing: {path}\nComplete data preparation and check the paper-era repository layout first.")


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("TORCH_HOME", str(ROOT / "environment" / "torch_cache"))
    env.setdefault("MPLCONFIGDIR", str(ROOT / "environment" / "matplotlib"))
    env.setdefault("MPLBACKEND", "Agg")  # Prevent AppKit/NSApplication abort in headless child processes.
    env.setdefault("PYTHONUNBUFFERED", "1")
    Path(env["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
    temporary_weight_link: Path | None = None
    if Path(args[1]).name == "nuc_train.py" and "--init-from-inc" in args:
        weight = ROOT / "environment" / "pretrained" / "resnet101-5d3b4d8f.pth"
        need(weight, "paper-era ImageNet ResNet-101 backbone weights; run scripts/download_pretrained_weights.sh")
        # nuc_train changes cwd to projects/placenta before model construction;
        # the paper-era loader uses model_dir="." relative to that directory.
        temporary_weight_link = PROJECT / weight.name
        if temporary_weight_link.is_symlink():
            if temporary_weight_link.resolve() != weight.resolve():
                raise SystemExit(f"Unexpected pretrained-weight symlink: {temporary_weight_link}")
            temporary_weight_link = None
        elif temporary_weight_link.exists():
            raise SystemExit(f"Refusing to overwrite {temporary_weight_link}; inspect the existing file first")
        else:
            temporary_weight_link.symlink_to(weight)
    experiment = args[args.index("--exp-name") + 1] if "--exp-name" in args else None
    if experiment is None:
        weight = args[args.index("--pre-trained") + 1] if "--pre-trained" in args else None
        weight = weight or (args[args.index("--pre-trained-path") + 1] if "--pre-trained-path" in args else None)
        experiment = Path(weight).stem if weight else Path(args[1]).stem
    log_path = log_dir / f"{experiment}.log"
    with log_path.open("a") as log:
        started_at = datetime.now(timezone.utc).isoformat()
        started_clock = time.monotonic()
        log.write(f"\nStarted UTC: {started_at}\nCommand: {' '.join(args)}\n")
        log.flush()
        try:
            process = subprocess.Popen(args, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            code = process.wait()
        finally:
            finished_at = datetime.now(timezone.utc).isoformat()
            log.write(f"Finished UTC: {finished_at}\nElapsed seconds: {time.monotonic() - started_clock:.2f}\n")
            log.flush()
            if temporary_weight_link is not None:
                temporary_weight_link.unlink(missing_ok=True)
    if code:
        raise subprocess.CalledProcessError(code, args)


def candidates(directory: Path, pattern: str) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and re.search(pattern, p.name, re.I))


def best_checkpoint(kind: str, exp_name: str) -> Path:
    directory = PROJECT / "results" / kind / exp_name
    files = list(directory.rglob("model_f1_*.pt" if kind == "nuclei" else "cell_model_accuracy_*.pt"))
    if not files and SMOKE:
        final_name = "nuclei_final_model.pt" if kind == "nuclei" else "cell_final_model.pt"
        files = list(directory.rglob(final_name))
    if not files:
        raise SystemExit(f"No best-validation checkpoint found under {directory}")
    def score(path: Path) -> float:
        match = re.search(r"_(\d+(?:\.\d+)?)\.pt$", path.name)
        return float(match.group(1)) if match else -1
    return max(files, key=lambda p: (score(p), p.stat().st_mtime))


def completed_smoke_checkpoint(kind: str, exp_name: str, final_name: str) -> Path | None:
    if not SMOKE:
        return None
    directory = PROJECT / "results" / kind / exp_name
    files = list(directory.rglob(final_name)) if directory.is_dir() else []
    if not files and directory.is_dir():
        pattern = "model_f1_*.pt" if kind == "nuclei" else "cell_model_accuracy_*.pt"
        files = list(directory.rglob(pattern))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def require_fresh_output(kind: str, exp_name: str) -> None:
    directory = PROJECT / "results" / kind / exp_name
    if directory.exists():
        raise SystemExit(
            f"Output already exists: {directory}. Refusing to repeat training and waste compute. "
            "The paper-era trainer saves weights but not optimizer state; inspect the checkpoint and logs before continuing manually."
        )


def train_nuclei() -> Path:
    need(PROJECT / "annotations" / "nuclei", "nuclei annotations")
    for dataset in ("hmc", "uot", "empty"):
        need(PROJECT / "datasets" / "nuclei" / dataset, f"nuclei dataset {dataset}")
    first_epochs, second_epochs = (1, 1) if SMOKE else (40, 60)
    phase_one, phase_two = f"repro-nuclei-backbone40{SUFFIX}", f"repro-nuclei-full60{SUFFIX}"
    loader_args = ["--num-workers", str(LOADER_WORKERS)] if LOADER_WORKERS is not None else []
    init = completed_smoke_checkpoint("nuclei", phase_one, "nuclei_final_model.pt")
    if init is None:
        require_fresh_output("nuclei", phase_one)
        run([str(PYTHON), "nuc_train.py", "--project-name", "placenta", "--exp-name", phase_one, "--annot-dir", "annotations/nuclei", "--dataset-names", "hmc", "--dataset-names", "uot", "--dataset-names", "empty", "--epochs", str(first_epochs), "--batch", "16", "--val-batch", "16", "--learning-rate", "0.0001", "--decay-gamma", "0.5", "--step-size", "20", *loader_args, "--init-from-inc", "--frozen"])
        init = best_checkpoint("nuclei", phase_one)
    if SMOKE:
        return init
    final_smoke = completed_smoke_checkpoint("nuclei", phase_two, "nuclei_final_model.pt")
    if final_smoke is not None:
        return final_smoke
    require_fresh_output("nuclei", phase_two)
    run([str(PYTHON), "nuc_train.py", "--project-name", "placenta", "--exp-name", phase_two, "--annot-dir", "annotations/nuclei", "--dataset-names", "hmc", "--dataset-names", "uot", "--dataset-names", "empty", "--epochs", str(second_epochs), "--batch", "16", "--val-batch", "16", "--learning-rate", "0.001", "--decay-gamma", "0.5", "--step-size", "20", *loader_args, "--pre-trained", str(init.relative_to(PROJECT)), "--no-frozen", "--no-init-from-inc"])
    final = best_checkpoint("nuclei", phase_two)
    if not SMOKE:
        run([str(PYTHON), "analysis/evaluation/evaluate_nuclei_model.py", "--project-name", "placenta", "--annot-dir", "annotations/nuclei", "--pre-trained", str(final.relative_to(PROJECT)), "--dataset-names", "hmc", "--dataset-names", "uot", "--dataset-names", "empty", "--use-test-set"])
    return final


def train_cells() -> Path:
    need(PROJECT / "annotations" / "cell_class", "cell annotations")
    for dataset in ("hmc", "uot", "nuh"):
        need(PROJECT / "datasets" / "cell_class" / dataset, f"cell dataset {dataset}")
    loader_args = ["--num-workers", str(LOADER_WORKERS)] if LOADER_WORKERS is not None else []
    common = ["--project-name", "placenta", "--organ-name", "placenta", "--annot-dir", "annotations/cell_class", "--dataset-names", "hmc", "--dataset-names", "uot", "--dataset-names", "nuh", "--batch", "600", "--val-batch", "600", "--learning-rate", "0.0001", "--decay-gamma", "0.5", "--step-size", "20", *loader_args]
    first_epochs, second_epochs = (1, 1) if SMOKE else (60, 100)
    phase_one, phase_two = f"repro-cell-imagenet60{SUFFIX}", f"repro-cell-full100{SUFFIX}"
    init = completed_smoke_checkpoint("cell_class", phase_one, "cell_final_model.pt")
    if init is None:
        require_fresh_output("cell_class", phase_one)
        run([str(PYTHON), "cell_train.py", *common, "--exp-name", phase_one, "--epochs", str(first_epochs), "--init-from-inc", "--frozen"])
        init = best_checkpoint("cell_class", phase_one)
    if SMOKE:
        return init
    final_smoke = completed_smoke_checkpoint("cell_class", phase_two, "cell_final_model.pt")
    if final_smoke is not None:
        return final_smoke
    require_fresh_output("cell_class", phase_two)
    run([str(PYTHON), "cell_train.py", *common, "--exp-name", phase_two, "--epochs", str(second_epochs), "--pre-trained", str(init.relative_to(PROJECT)), "--no-frozen", "--no-init-from-inc"])
    final = best_checkpoint("cell_class", phase_two)
    if not SMOKE:
        run([str(PYTHON), "analysis/evaluation/evaluate_cell_model.py", "--project-name", "placenta", "--organ-name", "placenta", "--annot-dir", "annotations/cell_class", "--pre-trained", str(final.relative_to(PROJECT)), "--dataset-names", "hmc", "--dataset-names", "uot", "--dataset-names", "nuh", "--use-test-set"])
    return final


def train_graph(args: argparse.Namespace) -> Path:
    need(PROJECT / "embeddings", "released graph embeddings")
    need(PROJECT / "annotations" / "graph", "graph tissue annotations")
    split_dir = PROJECT / "graph_splits"
    need(split_dir, "original graph split files")
    annotations = [PROJECT / "annotations" / "graph" / n for n in args.graph_annotations]
    for path in annotations:
        need(path, "graph annotation")
    for name in (args.val_split, args.test_split):
        need(split_dir / name, "original graph split")
    exp_name = f"repro-graph-clustergcn{SUFFIX}"
    require_fresh_output("graph", exp_name)
    command = [str(PYTHON), "graph_train.py", "--project-name", "placenta", "--organ-name", "placenta", "--exp-name", exp_name, "--run-ids", "1", "--run-ids", "2"]
    for path in annotations:
        command += ["--annot-tsvs", path.name]
    command += ["--val-patch-files", args.val_split, "--test-patch-files", args.test_split, "--layers", "16", "--hidden-units", "256", "--batch-size", "200", "--num-neighbours", "400", "--epochs", "1" if SMOKE else "2000", "--learning-rate", "0.001", "--weighted-loss", "--use-custom-weights"]
    run(command)
    out = PROJECT / "results" / "graph" / exp_name
    runs = sorted((p for p in out.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime) if out.exists() else []
    if not runs:
        raise SystemExit(f"No graph run directory found under {out}")
    run_dir = runs[-1]
    stats = run_dir / "graph_train_stats.csv"
    if not stats.is_file():
        raise SystemExit(f"Missing graph training metrics: {stats}")
    with stats.open(newline="") as f:
        rows = list(csv.DictReader(f))
    score_col = next((key for key in (rows[0].keys() if rows else []) if key.endswith("val_accuracy")), None)
    if not score_col:
        raise SystemExit(f"No validation accuracy in {stats}")
    valid = [(float(row[score_col]), int(float(row["Unnamed: 0"])) + 1) for row in rows if row.get(score_col)]
    if not valid:
        raise SystemExit(f"No validation scores in {stats}")
    _, best_epoch = max(valid)
    final = run_dir / f"{best_epoch}_graph_model.pt"
    need(final, "best-validation graph checkpoint")
    if not SMOKE:
        run([str(PYTHON), "analysis/evaluation/evaluate_graph_model.py", "--project-name", "placenta", "--organ-name", "placenta", "--pre-trained-path", str(final.relative_to(PROJECT)), "--run-id", "1", "--annot-tsv", args.graph_annotations[0], "--patch-files", args.test_split, "--plot"])
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "nuclei", "cell", "graph"), default="all")
    parser.add_argument("--graph-annotations", nargs=2, default=("wsi_1.tsv", "wsi_2.tsv"))
    parser.add_argument("--val-split", default="val_patches.csv")
    parser.add_argument("--test-split", default="test_patches.csv")
    parser.add_argument("--smoke-test", action="store_true", help="run one epoch per phase/stage; this is only an execution check, not a paper result")
    args = parser.parse_args()
    global SMOKE, SUFFIX
    SMOKE = args.smoke_test
    SUFFIX = "-smoke" if SMOKE else ""
    need(PYTHON, "isolated .venv Python; create it and install environment/requirements.txt")
    need(REPO / "COMMIT.txt", "paper-era source marker")
    # Extract public archives only after checksums and archive integrity pass.
    prepared = ROOT / "data" / "prepared"
    if not prepared.is_dir():
        raise SystemExit("Prepared data missing. First run: .venv/bin/python scripts/prepare_public_data.py")
    for name in ("annotations", "datasets", "embeddings", "graph_splits", "slides", "trained_models"):
        need(prepared / name, f"Zenodo folder {name}")
        link = PROJECT / name
        if not link.exists():
            link.symlink_to(prepared / name, target_is_directory=True)
    results_dir = PROJECT / "results"
    results_dir.mkdir(exist_ok=True)
    # v1.0.0 README says projects/placenta/embeddings, while its HDF5 reader
    # resolves paths under projects/placenta/results/embeddings.
    embeddings_alias = results_dir / "embeddings"
    if embeddings_alias.is_symlink() and embeddings_alias.resolve() != (prepared / "embeddings").resolve():
        raise SystemExit(f"Unexpected embeddings symlink target: {embeddings_alias} -> {embeddings_alias.resolve()}")
    if not embeddings_alias.exists() and not embeddings_alias.is_symlink():
        embeddings_alias.symlink_to(prepared / "embeddings", target_is_directory=True)
    elif embeddings_alias.exists() and not embeddings_alias.is_symlink():
        raise SystemExit(f"Refusing to use an existing non-symlink embedding directory at {embeddings_alias}; compare it with {prepared / 'embeddings'} first.")
    stages = {"nuclei": train_nuclei, "cell": train_cells, "graph": lambda: train_graph(args)}
    failures: list[str] = []
    for name in ("nuclei", "cell", "graph"):
        if args.stage not in ("all", name):
            continue
        print(f"\n=== {name.upper()} experiment ===", flush=True)
        if name == "graph":
            check = subprocess.run(
                [str(PYTHON), "-c", "import torch_cluster, torch_sparse"],
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            if check.returncode:
                message = (
                    "Graph stage blocked: paper-era PyG extensions torch_cluster and "
                    "torch_sparse are not importable. See environment/compatibility_notes.md.\n"
                    + check.stderr.strip()
                )
                if args.stage == "graph":
                    raise SystemExit(message)
                print(message, file=sys.stderr, flush=True)
                failures.append(f"graph: {message.splitlines()[0]}")
                continue
        try:
            stages[name]()
        except (Exception, SystemExit) as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"{name} stage failed: {message}", file=sys.stderr, flush=True)
            failures.append(f"{name}: {message}")
    stats = list((PROJECT / "results").glob("nuclei/*/*/nuclei_train_stats.csv"))
    stats += list((PROJECT / "results").glob("cell_class/*/*/cell_train_stats.csv"))
    stats += list((PROJECT / "results").glob("graph/*/*/graph_train_stats.csv"))
    if stats:
        try:
            run([str(PYTHON), str(ROOT / "scripts" / "plot_training_curves.py")])
        except (Exception, SystemExit) as exc:
            failures.append(f"loss plots: {type(exc).__name__}: {exc}")
    else:
        print("No training-stat CSVs exist yet; skipping loss plots.", flush=True)
    if failures:
        raise SystemExit("Some stages failed; independent stages were still attempted:\n- " + "\n- ".join(failures))
    print("Stages finished. Extract actual evaluator metrics before updating reports; do not enter target values as reproduced.")


if __name__ == "__main__":
    main()
