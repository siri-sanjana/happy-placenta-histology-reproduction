"""Run architectural novelty extension and ablation study on Stage 2 Cell Classification.

Novelty Components:
1. Residual Channel Attention Block (SE-style recalibration of 2048-dim histology features)
2. Mish Activation Function (smooth non-monotonic activation to preserve fine texture gradients)
3. Multi-Class Focal Loss (gamma=2.0 with class weighting to combat severe placenta cell imbalance)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add paper_compatible_repository to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper_compatible_repository"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from scipy.special import softmax
from sklearn.metrics import accuracy_score, top_k_accuracy_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

from happy.models.resnet import build_resnet
from happy.organs import get_organ
from happy.train.cell_train import setup_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_DIR = ROOT / "paper_compatible_repository" / "projects" / "placenta"
CACHE_FILE = ROOT / "data" / "prepared" / "cell_features_cache.pt"


# ==============================================================================
# 1. Feature Extraction / Caching
# ==============================================================================
def extract_or_load_features():
    os.chdir(str(PROJECT_DIR))
    if CACHE_FILE.exists():
        print(f"Loading cached features from {CACHE_FILE}...")
        return torch.load(CACHE_FILE, map_location="cpu")

    print("Extracting 2048-dim features from frozen ResNet-50 backbone...")
    organ = get_organ("placenta")
    HPs = namedtuple("HPs", "dataset_names batch")
    hp = HPs(["hmc", "uot", "nuh"], 100)

    # Load pretrained model
    model = build_resnet(out_features=len(organ.cells), depth=50)
    ckpt_path = ROOT / "data" / "prepared" / "trained_models" / "cell_model_accuracy_0.8472.pt"
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    # Define backbone extractor (up to avgpool)
    def extract_batch(img):
        feat = model.conv1(img)
        feat = model.bn1(feat)
        feat = model.relu(feat)
        feat = model.maxpool(feat)
        feat = model.layer1(feat)
        feat = model.layer2(feat)
        feat = model.layer3(feat)
        feat = model.layer4(feat)
        feat = model.avgpool(feat)
        return torch.flatten(feat, 1)

    # 1. Held-out test set
    test_loaders = setup_data(
        organ,
        PROJECT_DIR / "annotations" / "cell_class",
        hp,
        (224, 224),
        0,
        True,
        hp.batch,
        test_set=True,
    )

    test_features, test_labels, test_datasets = [], [], []
    with torch.no_grad():
        for dname in test_loaders:
            for batch in test_loaders[dname]:
                imgs = batch["img"].to(DEVICE).float()
                feats = extract_batch(imgs).cpu()
                labels = batch["annot"].cpu()
                test_features.append(feats)
                test_labels.append(labels)
                test_datasets.extend([dname] * len(labels))

    test_X = torch.cat(test_features, dim=0)
    test_y = torch.cat(test_labels, dim=0)

    # 2. Train and Val sets
    train_val_loaders = setup_data(
        organ,
        PROJECT_DIR / "annotations" / "cell_class",
        hp,
        (224, 224),
        0,
        True,
        hp.batch,
        test_set=False,
    )

    train_features, train_labels = [], []
    val_features, val_labels = [], []
    with torch.no_grad():
        for batch in train_val_loaders["train"]:
            imgs = batch["img"].to(DEVICE).float()
            feats = extract_batch(imgs).cpu()
            labels = batch["annot"].cpu()
            train_features.append(feats)
            train_labels.append(labels)

        # Validation loaders (hmc, uot, nuh)
        for dname in ["hmc", "uot", "nuh"]:
            if dname in train_val_loaders:
                for batch in train_val_loaders[dname]:
                    imgs = batch["img"].to(DEVICE).float()
                    feats = extract_batch(imgs).cpu()
                    labels = batch["annot"].cpu()
                    val_features.append(feats)
                    val_labels.append(labels)

    train_X = torch.cat(train_features, dim=0)
    train_y = torch.cat(train_labels, dim=0)
    val_X = torch.cat(val_features, dim=0)
    val_y = torch.cat(val_labels, dim=0)

    cache = {
        "train_X": train_X,
        "train_y": train_y,
        "val_X": val_X,
        "val_y": val_y,
        "test_X": test_X,
        "test_y": test_y,
        "test_datasets": test_datasets,
    }
    torch.save(cache, CACHE_FILE)
    print(f"Saved feature cache to {CACHE_FILE}: Train {len(train_y)}, Val {len(val_y)}, Test {len(test_y)}")
    return cache


# ==============================================================================
# 2. Architectures (Baseline & Novel Variants)
# ==============================================================================
class ResidualChannelAttention(nn.Module):
    """Channel Attention Module (Squeeze-and-Excitation recalibration) for 1D feature vectors."""
    def __init__(self, in_features: int = 2048, reduction: int = 16):
        super().__init__()
        reduced = max(in_features // reduction, 32)
        self.fc = nn.Sequential(
            nn.Linear(in_features, reduced, bias=False),
            nn.Mish(),
            nn.Linear(reduced, in_features, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.fc(x)
        return x * attn + x  # Residual connection


class CellClassificationHead(nn.Module):
    def __init__(
        self,
        in_features: int = 2048,
        num_classes: int = 11,
        use_attention: bool = False,
        activation: str = "relu",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attn = ResidualChannelAttention(in_features, reduction=16)

        Act = nn.Mish if activation == "mish" else nn.ReLU

        self.mlp = nn.Sequential(
            nn.Linear(in_features, 768),
            Act(),
            nn.Dropout(p=dropout),
            nn.Linear(768, 384),
            Act(),
            nn.Dropout(p=dropout),
            nn.Linear(384, 64),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attention:
            x = self.attn(x)
        return self.mlp(x)


# ==============================================================================
# 3. Loss Functions
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        p = torch.exp(-ce_loss)
        focal_loss = ((1.0 - p) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ==============================================================================
# 4. Evaluation Helper
# ==============================================================================
def evaluate_model(model: nn.Module, test_X: torch.Tensor, test_y: torch.Tensor, organ_cells):
    model.eval()
    with torch.no_grad():
        logits = model(test_X.to(DEVICE)).cpu()
        probs = F.softmax(logits, dim=-1).numpy()
        preds = logits.argmax(dim=-1).numpy()
        y_true = test_y.numpy()

    labels = [c.id for c in organ_cells]
    acc = accuracy_score(y_true, preds)
    top2 = top_k_accuracy_score(y_true, probs, k=2, labels=labels)
    roc_auc = roc_auc_score(y_true, probs, average="macro", multi_class="ovo", labels=labels)

    return {
        "accuracy": acc,
        "top_2_accuracy": top2,
        "macro_roc_auc": roc_auc,
        "logits": logits,
        "preds": preds,
    }


# ==============================================================================
# 5. Training Loop
# ==============================================================================
def train_head(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    epochs: int = 40,
    lr: float = 1e-4,
) -> nn.Module:
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(DEVICE)
                vout = model(vx)
                val_preds.extend(vout.argmax(dim=-1).cpu().tolist())
                val_targets.extend(vy.tolist())
        val_acc = accuracy_score(val_targets, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ==============================================================================
# 6. Main Experiment & Ablation Study
# ==============================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    cache = extract_or_load_features()
    train_X, train_y = cache["train_X"], cache["train_y"]
    val_X, val_y = cache["val_X"], cache["val_y"]
    test_X, test_y = cache["test_X"], cache["test_y"]

    organ = get_organ("placenta")
    organ_cells = organ.cells
    num_classes = len(organ_cells)

    # Inverse-frequency class weights for training
    classes, counts = torch.unique(train_y, return_counts=True)
    weights = torch.ones(num_classes, dtype=torch.float32)
    for c, cnt in zip(classes, counts):
        weights[c] = 1.0 / (float(cnt) ** 0.5)
    weights = (weights / weights.sum() * num_classes).to(DEVICE)

    train_ds = TensorDataset(train_X, train_y)
    val_ds = TensorDataset(val_X, val_y)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    print("\n--- 1. Evaluating Authors' Released Pretrained Baseline Head ---")
    orig_model = build_resnet(out_features=num_classes, depth=50)
    ckpt_path = ROOT / "data" / "prepared" / "trained_models" / "cell_model_accuracy_0.8472.pt"
    orig_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    orig_head = orig_model.fc.to(DEVICE)
    baseline_metrics = evaluate_model(orig_head, test_X, test_y, organ_cells)
    print(f"Baseline (Replicated): Accuracy={baseline_metrics['accuracy']:.4f}, Top-2={baseline_metrics['top_2_accuracy']:.4f}, Macro ROC-AUC={baseline_metrics['macro_roc_auc']:.4f}")

    # Experiments:
    configs = [
        {
            "name": "Baseline (Re-trained MLP)",
            "use_attention": False,
            "activation": "relu",
            "loss": "ce",
            "notes": "Standard MLP head, ReLU activation, Weighted Cross-Entropy Loss",
        },
        {
            "name": "+ Mish Activation",
            "use_attention": False,
            "activation": "mish",
            "loss": "ce",
            "notes": "Replaces ReLU with smooth non-monotonic Mish activation",
        },
        {
            "name": "+ Focal Loss",
            "use_attention": False,
            "activation": "relu",
            "loss": "focal",
            "notes": "Replaces CE with Focal Loss (gamma=2.0) to focus on hard/minority cells",
        },
        {
            "name": "+ Channel Attention (SE)",
            "use_attention": True,
            "activation": "relu",
            "loss": "ce",
            "notes": "Adds residual Squeeze-and-Excitation channel attention before MLP",
        },
        {
            "name": "Proposed Full Model (Attention + Mish + Focal)",
            "use_attention": True,
            "activation": "mish",
            "loss": "focal",
            "notes": "Residual Channel Attention + Mish activation + Multi-class Focal Loss",
        },
    ]

    results = [
        {
            "Model Configuration": "Baseline (Authors Released / Replicated)",
            "Accuracy": f"{baseline_metrics['accuracy'] * 100:.2f}%",
            "Top-2 Accuracy": f"{baseline_metrics['top_2_accuracy'] * 100:.2f}%",
            "Macro ROC-AUC": f"{baseline_metrics['macro_roc_auc']:.4f}",
            "Diff vs Baseline (Acc)": "+0.00%",
            "Description": "Original ResNet-50 head (ReLU + Cross-Entropy)",
        }
    ]

    print("\n--- Running Ablation Study Models ---")
    for cfg in configs:
        print(f"Training: {cfg['name']}...")
        torch.manual_seed(42)
        model = CellClassificationHead(
            in_features=2048,
            num_classes=num_classes,
            use_attention=cfg["use_attention"],
            activation=cfg["activation"],
            dropout=0.3,
        )
        if cfg["loss"] == "focal":
            criterion = FocalLoss(gamma=2.0, weight=weights)
        else:
            criterion = nn.CrossEntropyLoss(weight=weights)

        trained_model = train_head(model, train_loader, val_loader, criterion, epochs=40, lr=2e-4)
        m = evaluate_model(trained_model, test_X, test_y, organ_cells)
        diff_acc = (m["accuracy"] - baseline_metrics["accuracy"]) * 100

        print(f"  -> Acc: {m['accuracy']*100:.2f}%, Top-2: {m['top_2_accuracy']*100:.2f}%, AUC: {m['macro_roc_auc']:.4f} (Diff: {diff_acc:+.2f}%)")

        results.append({
            "Model Configuration": cfg["name"],
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Top-2 Accuracy": f"{m['top_2_accuracy'] * 100:.2f}%",
            "Macro ROC-AUC": f"{m['macro_roc_auc']:.4f}",
            "Diff vs Baseline (Acc)": f"{diff_acc:+.2f}%",
            "Description": cfg["notes"],
        })

    # Save results to CSV and Markdown
    df_results = pd.DataFrame(results)
    csv_path = ROOT / "reports" / "novelty_cell_classification.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved novelty comparison table to {csv_path}")

    # Generate Markdown table
    md_lines = [
        "# Novelty Architectural Extension: Residual Channel Attention & Loss Optimization",
        "",
        "## Summary of Results (Held-out Test Split: 2,743 cells across HMC, UOT, NUH)",
        "",
        df_results.to_markdown(index=False),
        "",
    ]
    md_path = ROOT / "reports" / "novelty_cell_classification.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Saved Markdown report to {md_path}")


if __name__ == "__main__":
    main()
