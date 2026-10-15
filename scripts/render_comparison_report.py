#!/usr/bin/env python3
"""Render the paper-versus-reproduction Markdown report from the measured-results CSV."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "paper_vs_reproduction.csv"
TARGET = ROOT / "reports" / "paper_vs_reproduction.md"


def cell(value: str | None) -> str:
    value = (value or "").strip()
    return value.replace("|", "\\|") if value else "—"


def main() -> None:
    with SOURCE.open(newline="") as f:
        rows = list(csv.DictReader(f))
    lines = [
        "# Paper versus reproduction",
        "",
        "Reproduced values are entered only from the original test evaluators. Smoke checkpoints and validation metrics are excluded.",
        "",
        "| Stage | Metric | Published | Reproduced | Absolute difference | Test split and sample count | Status |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        sample = row["Evaluated samples"]
        split = row["Evaluation split"]
        lines.append(
            "| {stage} | {metric} | {published} | {reproduced} | {difference} | {split}; {sample} | {status} |".format(
                stage=cell(row["Experiment"]),
                metric=cell(row["Metric"]),
                published=cell(row["Published value"]),
                reproduced=cell(row["Reproduced value"]),
                difference=cell(row["Absolute difference"]),
                split=cell(split),
                sample=cell(sample),
                status=cell(row["Reproduction status"]),
            )
        )
    lines += ["", "## Protocol and status", ""]
    explanations = []
    for row in rows:
        explanation = row["Explanation of discrepancies"].strip()
        if explanation and explanation not in explanations:
            explanations.append(explanation)
    lines.extend(f"- {cell(explanation)}" for explanation in explanations)
    lines += [
        "",
        "The published tissue test count differs by location: the abstract and Supplementary Table 7 sum to 179,095 nodes; the Results text and repository README state 149,425. The actual released-split evaluator count must be recorded before claiming a direct comparison.",
        "",
        "The paper's Methods/README describe COCO initialisation for nuclei detection, while paper-era v1.0.0 code loads an ImageNet ResNet-101 backbone and initializes detector heads. This implementation discrepancy remains documented in `references/reproduction_reference.md`.",
        "",
        "No exact reproduction is claimed: neither full paper schedule nor either original held-out test evaluation ran. The 59B.svs transfer remains irrelevant to these two released-patch experiments.",
        "",
    ]
    TARGET.write_text("\n".join(lines))
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
