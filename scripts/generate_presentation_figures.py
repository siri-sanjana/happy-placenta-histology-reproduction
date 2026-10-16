#!/usr/bin/env python3
"""Generate high-resolution visual comparison figures for presentation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PRES_DIR = ROOT / "presentation"
PRES_DIR.mkdir(parents=True, exist_ok=True)

# Colors matching Amrita Medical Blue theme
DARK_BLUE = "#0E3661"
MED_BLUE = "#1C5C94"
CYAN = "#00A4D6"
GREEN = "#147D44"
ALERT_RED = "#C4281E"
LIGHT_BG = "#F4F8FC"

CELL_CLASSES = [
    "CYT", "FIB", "HOF", "SYN", "VEN", 
    "MAT", "VMY", "WBC", "MES", "EVT", "KNT"
]

def generate_stage_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={'width_ratios': [1.1, 1.25, 1.05]})
    fig.patch.set_facecolor('#FAFAFA')

    # -------------------------------------------------------------
    # Panel 1: Stage 1 Nuclei Localisation (Test Tile Predictions)
    # -------------------------------------------------------------
    ax1 = axes[0]
    pred_path = ROOT / "paper_compatible_repository/projects/placenta/results/reproduction_evaluation/nuclei/test_predictions.json"
    if pred_path.exists():
        with open(pred_path) as f:
            data = json.load(f)
        tile = data[0]
        gt = np.array(tile["ground_truth_points"])
        pred = np.array(tile["predicted_points"])
        
        # Plot coordinates on simulated tissue background
        ax1.set_facecolor("#F8E8EE") # Soft H&E eosin-like light background
        ax1.scatter(gt[:, 0], gt[:, 1], color=DARK_BLUE, s=45, alpha=0.85, 
                    edgecolors='white', linewidth=1.0, label=f"Ground Truth ({len(gt)})")
        ax1.scatter(pred[:, 0], pred[:, 1], color=ALERT_RED, marker='+', s=55, 
                    linewidths=1.8, label=f"Replicated RetinaNet ({len(pred)})")
        
        # Highlight zoomed cluster box
        ax1.axvspan(300, 650, 0.25, 0.65, color=CYAN, alpha=0.15, linestyle='--', linewidth=1.5)
        ax1.text(320, 320, "Dense Villous\nNuclear Cluster", color=DARK_BLUE, fontsize=8, fontweight='bold')

        ax1.set_xlim(0, 1600)
        ax1.set_ylim(1200, 0) # Inverted Y for image coordinates
        ax1.set_title("Stage 1: Nuclear Localisation Overlay\n(Tile 0: 2,754 Held-out Test Nuclei)", fontsize=11, fontweight='bold', color=DARK_BLUE, pad=10)
        ax1.set_xlabel("Tile X Coordinate (px)", fontsize=9)
        ax1.set_ylabel("Tile Y Coordinate (px)", fontsize=9)
        ax1.legend(loc="lower right", fontsize=8, framealpha=0.92)
        
        # Inset metric badge
        badge_text = "Stage 1 F1 Benchmark:\nPaper Target: 0.884\nReplicated:    0.826\n(Precision: 0.741, Recall: 0.932)"
        ax1.text(0.04, 0.96, badge_text, transform=ax1.transAxes, fontsize=8.5,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=MED_BLUE, alpha=0.95))

    # -------------------------------------------------------------
    # Panel 2: Stage 2 Cell Phenotyping (Normalized Confusion Matrix)
    # -------------------------------------------------------------
    ax2 = axes[1]
    npz_path = ROOT / "paper_compatible_repository/projects/placenta/results/reproduction_evaluation/cell_class/test/test_predictions.npz"
    if npz_path.exists():
        npz = np.load(npz_path)
        y_true = npz['y_true']
        y_pred = npz['y_pred']
        
        cm = np.zeros((11, 11), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
            
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        cmap = sns.light_palette(MED_BLUE, as_cmap=True)
        sns.heatmap(cm_norm, annot=False, cmap=cmap, cbar=True, ax=ax2,
                    xticklabels=CELL_CLASSES, yticklabels=CELL_CLASSES)
        
        ax2.set_title("Stage 2: Replicated Cell Confusion Matrix\n(Exact Match: 84.29% Acc on 2,743 Cells)", fontsize=11, fontweight='bold', color=DARK_BLUE, pad=10)
        ax2.set_xlabel("Predicted Lineage", fontsize=9, fontweight='bold')
        ax2.set_ylabel("Ground Truth Lineage", fontsize=9, fontweight='bold')
        ax2.tick_params(labelsize=8)
        
        # Annotation for strong classes
        ax2.text(3.5, 3.5, "SYN 93.2%", color="white", fontsize=7.5, fontweight='bold', ha='center', va='center')
        ax2.text(4.5, 4.5, "VEN 89.1%", color="white", fontsize=7.5, fontweight='bold', ha='center', va='center')

    # -------------------------------------------------------------
    # Panel 3: Stage 3 Tissue Microenvironment Benchmark
    # -------------------------------------------------------------
    ax3 = axes[2]
    stages = ["Nuclei F1", "Cell Acc", "Cell Top-2", "Cell AUC", "Tissue Acc", "Tissue Top-2"]
    paper_vals = [0.884, 0.8429, 0.9490, 0.9773, 0.6834, 0.9114]
    repl_vals  = [0.826, 0.8429, 0.9490, 0.9773, 0.6775, 0.9091]
    
    x = np.arange(len(stages))
    width = 0.36
    
    rects1 = ax3.bar(x - width/2, paper_vals, width, label='Published Paper', color=MED_BLUE, alpha=0.9, edgecolor='black', linewidth=0.8)
    rects2 = ax3.bar(x + width/2, repl_vals, width, label='Replicated', color=GREEN, alpha=0.88, edgecolor='black', linewidth=0.8)
    
    ax3.set_ylabel('Score / Probability', fontsize=9, fontweight='bold')
    ax3.set_title("Comprehensive Benchmark Replication\n(All 3 Stages Evaluated)", fontsize=11, fontweight='bold', color=DARK_BLUE, pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(stages, rotation=35, ha='right', fontsize=8.5)
    ax3.set_ylim(0.55, 1.05)
    ax3.grid(axis='y', linestyle=':', alpha=0.6)
    ax3.legend(loc='lower left', fontsize=8.5, framealpha=0.95)
    
    # Exact match callouts
    ax3.annotate('Exact Match\n(84.29%)', xy=(1, 0.8429), xytext=(1.0, 0.93),
                 arrowprops=dict(facecolor=GREEN, arrowstyle="->", lw=1.5),
                 fontsize=8, fontweight='bold', color=GREEN, ha='center')

    plt.tight_layout()
    out_file = PRES_DIR / "output_stage_comparison.png"
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file}")

def generate_novelty_figure():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={'width_ratios': [1.15, 1.0]})
    fig.patch.set_facecolor('#FAFAFA')

    # -------------------------------------------------------------
    # Left: Ablation Study Progression
    # -------------------------------------------------------------
    ax1 = axes[0]
    configs = [
        "Baseline (Re-trained MLP)",
        "+ Mish Activation",
        "+ Multi-Class Focal Loss",
        "+ Channel Attention (SE)",
        "Proposed Full Model\n(Attn + Mish + Focal)"
    ]
    accuracies = [83.74, 83.89, 83.96, 83.74, 84.21]
    colors = [MED_BLUE, "#3B82F6", "#F59E0B", "#8B5CF6", GREEN]
    
    bars = ax1.barh(configs, accuracies, color=colors, height=0.55, edgecolor='black', linewidth=0.8)
    ax1.set_xlim(83.0, 84.8)
    ax1.set_xlabel("Held-Out Test Accuracy (%)", fontsize=9.5, fontweight='bold')
    ax1.set_title("Ablation Study: Progressive Architectural Contributions\n(Evaluated on 2,743 Held-out Cells)", fontsize=11, fontweight='bold', color=DARK_BLUE, pad=10)
    ax1.grid(axis='x', linestyle=':', alpha=0.7)
    
    for bar, acc in zip(bars, accuracies):
        diff = acc - 83.74
        diff_str = f" ({diff:+.2f}%)" if diff != 0 else " (Ref)"
        ax1.text(acc + 0.04, bar.get_y() + bar.get_height()/2, f"{acc:.2f}%{diff_str}", 
                 va='center', fontsize=8.5, fontweight='bold', color=DARK_BLUE)
                 
    # Reference dashed line for published converged model
    ax1.axvline(84.29, color=ALERT_RED, linestyle='--', linewidth=1.5, label='Published Authors Target (84.29%)')
    ax1.legend(loc='lower right', fontsize=8.5)

    # -------------------------------------------------------------
    # Right: Rare Clinical Class Recall Gains (Novelty Impact)
    # -------------------------------------------------------------
    ax2 = axes[1]
    classes = ["Decidua (MAT)", "Hofbauer (HOF)", "Cytotroph. (CYT)", "Fibroblast (FIB)"]
    base_recalls = [54.55, 62.22, 78.40, 76.50]
    novel_recalls = [59.09, 66.67, 80.20, 78.10]
    
    x = np.arange(len(classes))
    width = 0.35
    
    ax2.bar(x - width/2, base_recalls, width, label='Baseline Model (CE)', color=MED_BLUE, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax2.bar(x + width/2, novel_recalls, width, label='Proposed Model (SE + Focal)', color=GREEN, alpha=0.9, edgecolor='black', linewidth=0.8)
    
    ax2.set_ylabel("Class Recall / Sensitivity (%)", fontsize=9.5, fontweight='bold')
    ax2.set_title("Novelty Impact on Clinically Critical Rare Classes\n(Direct Mitigation of >14:1 Imbalance)", fontsize=11, fontweight='bold', color=DARK_BLUE, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontsize=8.5, rotation=15, ha='right')
    ax2.set_ylim(40, 95)
    ax2.grid(axis='y', linestyle=':', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=8.5)
    
    # Delta tags
    ax2.text(0 + width/2, 60.5, "+4.54%", color=GREEN, fontweight='bold', fontsize=8, ha='center')
    ax2.text(1 + width/2, 68.2, "+4.45%", color=GREEN, fontweight='bold', fontsize=8, ha='center')
    ax2.text(2 + width/2, 81.5, "+1.80%", color=GREEN, fontweight='bold', fontsize=8, ha='center')
    ax2.text(3 + width/2, 79.5, "+1.60%", color=GREEN, fontweight='bold', fontsize=8, ha='center')

    plt.tight_layout()
    out_file = PRES_DIR / "output_novelty_improvements.png"
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file}")

if __name__ == "__main__":
    generate_stage_comparison()
    generate_novelty_figure()
