#!/usr/bin/env python3
"""Plot loss columns from paper-era HAPPY training-stat CSV files."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1] / "paper_compatible_repository")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "loss_curves")
    args = parser.parse_args()
    project = args.repo / "projects" / "placenta"
    files = list(project.glob("results/nuclei/*/*/nuclei_train_stats.csv"))
    files += list(project.glob("results/cell_class/*/*/cell_train_stats.csv"))
    files += list(project.glob("results/graph/*/*/graph_train_stats.csv"))
    if not files:
        raise SystemExit(f"No original training-stat CSVs found under {project / 'results'}")
    args.output.mkdir(parents=True, exist_ok=True)
    plotted = 0
    for path in sorted(files):
        frame = pd.read_csv(path, index_col=0)
        columns = [c for c in frame.columns if "loss" in str(c).lower()]
        columns = [c for c in columns if frame[c].notna().any()]
        if not columns:
            continue
        plt.figure(figsize=(8, 5))
        for column in columns:
            plt.plot(frame.index, frame[column], label=str(column))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{path.parent.parent.name} — {path.parent.name}")
        plt.legend()
        plt.tight_layout()
        target = args.output / f"{path.parent.parent.name}-{path.parent.name}-loss.png"
        plt.savefig(target, dpi=160)
        plt.close()
        plotted += 1
    if not plotted:
        raise SystemExit("Training-stat files were found, but none contained recorded loss values")
    print(f"Saved {plotted} loss plots to {args.output}")


if __name__ == "__main__":
    main()
