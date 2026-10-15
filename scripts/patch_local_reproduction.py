#!/usr/bin/env python3
"""Apply small local-only plotting/output fixes to the paper-era checkout.

The official current repository remains untouched. Training, model, split,
optimizer, and metric calculations are not changed by these output hooks.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "paper_compatible_repository"
MARKER = "# HAPPY_LOCAL_OUTPUT_PATCH"


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if new in source:
        return
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one patch point in {path.relative_to(ROOT)}; found {count}: {old[:100]!r}")
    updated = source.replace(old, new, 1)
    if MARKER not in source:
        updated = MARKER + "\n" + updated
    path.write_text(updated)


def main() -> None:
    runner = ROOT / "scripts" / "run_public_reproduction.py"
    replace_once(
        runner,
        '    env.setdefault("MPLCONFIGDIR", str(ROOT / "environment" / "matplotlib"))\n',
        '    env.setdefault("MPLCONFIGDIR", str(ROOT / "environment" / "matplotlib"))\n'
        '    env.setdefault("MPLBACKEND", "Agg")  # Prevent AppKit/NSApplication abort in headless child processes.\n',
    )

    cell_train = REPO / "happy" / "train" / "cell_train.py"
    replace_once(
        cell_train,
        '        cm = get_cell_confusion_matrix(organ, pred[dataset], truth[dataset])\n'
        '        logger.log_confusion_matrix(cm, dataset, run_path)\n',
        '        cm = get_cell_confusion_matrix(organ, pred[dataset], truth[dataset])\n'
        '        try:\n'
        '            logger.log_confusion_matrix(cm, dataset, run_path)\n'
        '        except Exception as exc:\n'
        '            print(f"Optional confusion-matrix plot failed for {dataset}: {type(exc).__name__}: {exc}", flush=True)\n',
    )

    cell_eval = REPO / "analysis" / "evaluation" / "evaluate_cell_model.py"
    replace_once(
        cell_eval,
        "from pathlib import Path\n",
        "from pathlib import Path\nimport json\nimport numpy as np\n",
    )
    replace_once(cell_eval, "import os\n", "import os\nimport platform\n")
    replace_once(
        cell_eval,
        '        (224, 224),\n        3,\n',
        '        (224, 224),\n        0 if platform.system() == "Darwin" else 3,\n',
    )
    replace_once(
        cell_eval,
        '        print(f"ROC AUC macro: {roc_auc:.6f}")\n\n'
        '        if plot_cm:\n'
        '            cell_mapping = {cell.id: cell.name for cell in organ.cells}\n'
        '            _plot_confusion_matrix(\n'
        '                organ,\n'
        '                predictions[dataset_name],\n'
        '                ground_truth[dataset_name],\n'
        '                dataset_name,\n'
        '            )\n',
        '        print(f"ROC AUC macro: {roc_auc:.6f}")\n'
        '        # Persist exact test targets, predictions and raw logits before any optional plot.\n'
        '        evaluation_dir = project_dir / "results" / "reproduction_evaluation" / "cell_class" / ("test" if use_test_set else "validation")\n'
        '        evaluation_dir.mkdir(parents=True, exist_ok=True)\n'
        '        np.savez_compressed(evaluation_dir / f"{dataset_name}_predictions.npz", y_true=np.asarray(ground_truth[dataset_name]), y_pred=np.asarray(predictions[dataset_name]), logits=np.asarray(outs[dataset_name]))\n'
        '        metrics = {"accuracy": float(accuracy), "top2_accuracy": float(top_2_accuracy), "macro_roc_auc_ovo": float(roc_auc), "n": len(ground_truth[dataset_name]), "classes": cell_mapping}\n'
        '        (evaluation_dir / f"{dataset_name}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\\n")\n'
        '        if plot_cm:\n'
        '            cell_mapping = {cell.id: cell.name for cell in organ.cells}\n'
        '            try:\n'
        '                _plot_confusion_matrix(organ, predictions[dataset_name], ground_truth[dataset_name], dataset_name)\n'
        '            except Exception as exc:\n'
        '                print(f"Optional confusion-matrix plot failed for {dataset_name}: {type(exc).__name__}: {exc}", flush=True)\n',
    )
    replace_once(
        cell_eval,
        '    plot_confusion_matrix(\n'
        '        cm,\n'
        '        dataset_name,\n'
        '        Path(f"../../analysis/evaluation/plots/"),\n',
        '    output_dir = Path("results/reproduction_evaluation/cell_class") / dataset_name\n'
        '    output_dir.mkdir(parents=True, exist_ok=True)\n'
        '    plot_confusion_matrix(\n'
        '        cm,\n'
        '        dataset_name,\n'
        '        output_dir,\n',
    )

    point_eval = REPO / "happy" / "train" / "point_eval.py"
    replace_once(
        point_eval,
        "from scipy.spatial import distance_matrix\n",
        "from scipy.spatial import distance_matrix\nfrom pathlib import Path\nimport json\n",
    )
    replace_once(
        point_eval,
        "    num_false_negative = 0\n    for data in tqdm(dataloader):\n",
        "    num_false_negative = 0\n    prediction_records = []\n    image_index = 0\n    for data in tqdm(dataloader):\n",
    )
    replace_once(
        point_eval,
        "        ground_truth_points = convert_boxes_to_points(gt_predictions)\n\n"
        "        # For batches with all empty images\n",
        "        ground_truth_points = convert_boxes_to_points(gt_predictions)\n"
        '        empty_image = bool(data["annot"][0][0][-1].numpy() == -1)\n'
        '        prediction_records.append({"image_index": image_index, "empty_image": empty_image, "ground_truth_points": [] if empty_image else np.asarray(ground_truth_points).reshape(-1, 2).tolist(), "predicted_points": np.asarray(predicted_points).reshape(-1, 2).tolist()})\n'
        "        image_index += 1\n\n        # For batches with all empty images\n",
    )
    replace_once(
        point_eval,
        '        if data["annot"][0][0][-1].numpy() == -1:\n',
        "        if empty_image:\n",
    )
    replace_once(
        point_eval,
        "    # If all images in the dataset are empty\n"
        "    if num_true_positive == 0 and num_false_negative == 0:\n",
        '    output_path = globals().get("HAPPY_LOCAL_PREDICTION_PATH")\n'
        "    if output_path is not None:\n"
        "        output_path = Path(output_path)\n"
        "        output_path.parent.mkdir(parents=True, exist_ok=True)\n"
        '        output_path.write_text(json.dumps(prediction_records) + "\\n")\n'
        "    # If all images in the dataset are empty\n"
        "    if num_true_positive == 0 and num_false_negative == 0:\n",
    )

    nuc_eval = REPO / "analysis" / "evaluation" / "evaluate_nuclei_model.py"
    replace_once(nuc_eval, "from collections import namedtuple\n", "from collections import namedtuple\nimport json\nfrom pathlib import Path\n")
    replace_once(nuc_eval, "import os\n", "import os\nimport platform\n")
    replace_once(
        nuc_eval,
        '    dataloaders = setup_data(\n        project_dir / annot_dir, hp, multiple_val_sets, 3, 1, use_test_set\n    )\n',
        '    dataloaders = setup_data(\n        project_dir / annot_dir, hp, multiple_val_sets, 0 if platform.system() == "Darwin" else 3, 1, use_test_set\n    )\n',
    )
    replace_once(
        nuc_eval,
        "    mean_precision = {}\n    mean_recall = {}\n    mean_f1 = {}\n    num_empty_predictions = {}\n",
        '    mean_precision = {}\n    mean_recall = {}\n    mean_f1 = {}\n    num_empty_predictions = {}\n'
        '    evaluation_dir = project_dir / "results" / "reproduction_evaluation" / "nuclei"\n'
        "    evaluation_dir.mkdir(parents=True, exist_ok=True)\n",
    )
    replace_once(
        nuc_eval,
        "    for dataset_name in dataloaders:\n"
        "        precision, recall, f1, num_empty = evaluate_points_over_dataset(\n",
        "    for dataset_name in dataloaders:\n"
        "        from happy.train import point_eval\n"
        '        point_eval.HAPPY_LOCAL_PREDICTION_PATH = evaluation_dir / f"{dataset_name}_predictions.json"\n'
        "        precision, recall, f1, num_empty = evaluate_points_over_dataset(\n",
    )
    replace_once(
        nuc_eval,
        "    try:\n"
        '        print(\n            f"Number of Predictions in empty images: {num_empty_predictions[\'empty\']}"\n'
        "        )\n"
        "    except KeyError:\n"
        "        pass\n",
        "    (evaluation_dir / \"metrics.json\").write_text(json.dumps({name: {\"precision\": mean_precision[name], \"recall\": mean_recall[name], \"f1\": mean_f1[name], \"false_positives_on_empty_images\": int(num_empty_predictions[name])} for name in mean_f1}, indent=2) + \"\\n\")\n"
        "    try:\n"
        '        print(\n            f"Number of Predictions in empty images: {num_empty_predictions[\'empty\']}"\n'
        "        )\n"
        "    except KeyError:\n"
        "        pass\n",
    )
    print("Applied local-only Agg/optional-plot and evaluation-output patches; model/config/metric formulas unchanged.")


if __name__ == "__main__":
    main()
