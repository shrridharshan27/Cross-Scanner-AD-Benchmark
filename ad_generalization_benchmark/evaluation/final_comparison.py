"""Final Benchmark Comparison and Visualization Pipeline.

Aggregates performance metrics across all methods (Baseline LR, Baseline SVM, Baseline MLP,
ComBat+MLP, DANN) across seeds [42, 123, 456]. Generates publication-ready ROC curves,
AUROC grouped comparison bar charts, generalization gap plots, and final comparison tables.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import torch

from ad_generalization_benchmark.data.loaders.oasis1_loader import load_oasis1
from ad_generalization_benchmark.features.align_features import apply_scaler, load_scaler
from ad_generalization_benchmark.models.mlp import MLPClassifier
from ad_generalization_benchmark.training.train_dann import DANNArchitecture
from neuroCombat import neuroCombatFromTraining

FEATURE_ORDER = ["nWBV", "eTIV", "age", "sex"]
SPANISH_MAP = {
    "brain_vol_ratio": "nWBV",
    "icv_estimate": "eTIV",
    "age": "age",
    "sex": "sex",
}


def load_all_benchmark_results(results_dir: Path, seeds: List[int]) -> Dict[str, Dict[str, Any]]:
    """Load and aggregate metrics across seeds for all 5 methods."""
    methods = {
        "LR (baseline)": {"oasis_aucs": [], "sp_aucs": [], "gaps": []},
        "SVM (baseline)": {"oasis_aucs": [], "sp_aucs": [], "gaps": []},
        "MLP (baseline)": {"oasis_aucs": [], "sp_aucs": [], "gaps": []},
        "ComBat+MLP": {"oasis_aucs": [], "sp_aucs": [], "gaps": []},
        "DANN": {"oasis_aucs": [], "sp_aucs": [], "gaps": []},
    }

    for s in seeds:
        # 1. Baselines internal & external
        with open(results_dir / f"internal_eval_seed{s}.json", "r", encoding="utf-8") as f:
            int_data = json.load(f)
        with open(results_dir / f"external_trackA_seed{s}.json", "r", encoding="utf-8") as f:
            ext_data = json.load(f)

        for key, m_name in [("lr", "LR (baseline)"), ("svm", "SVM (baseline)"), ("mlp", "MLP (baseline)")]:
            oasis_auc = int_data[key]["auroc"]
            sp_auc = ext_data[key]["spanish_auroc"]
            gap = ext_data[key]["gap_auroc"]
            methods[m_name]["oasis_aucs"].append(oasis_auc)
            methods[m_name]["sp_aucs"].append(sp_auc)
            methods[m_name]["gaps"].append(gap)

        # 2. ComBat
        with open(results_dir / f"combat_eval_seed{s}.json", "r", encoding="utf-8") as f:
            combat_data = json.load(f)
        methods["ComBat+MLP"]["oasis_aucs"].append(combat_data["internal"]["auroc"])
        methods["ComBat+MLP"]["sp_aucs"].append(combat_data["external"]["auroc"])
        methods["ComBat+MLP"]["gaps"].append(combat_data["generalization_gap"])

        # 3. DANN
        with open(results_dir / f"dann_eval_seed{s}.json", "r", encoding="utf-8") as f:
            dann_data = json.load(f)
        methods["DANN"]["oasis_aucs"].append(dann_data["internal"]["auroc"])
        methods["DANN"]["sp_aucs"].append(dann_data["external"]["auroc"])
        methods["DANN"]["gaps"].append(dann_data["generalization_gap"])

    summary = {}
    for m, d in methods.items():
        summary[m] = {
            "oasis_mean": float(np.mean(d["oasis_aucs"])),
            "oasis_std": float(np.std(d["oasis_aucs"])),
            "sp_mean": float(np.mean(d["sp_aucs"])),
            "sp_std": float(np.std(d["sp_aucs"])),
            "gap_mean": float(np.mean(d["gaps"])),
            "gap_std": float(np.std(d["gaps"])),
        }
    return summary


def generate_comparison_plots(
    repo_dir: Path,
    plots_dir: Path,
    summary_data: Dict[str, Dict[str, Any]],
    seed: int = 42,
) -> None:
    """Generate all 4 required benchmark plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Load data for ROC plotting
    df_oasis, _ = load_oasis1(verbose=False)
    sp_cache = repo_dir / "features" / "feature_cache" / "spanish_features.csv"
    df_sp = pd.read_csv(sp_cache)

    splits_dir = repo_dir / "splits" / "saved"
    models_dir = repo_dir / "models" / "saved"
    scalers_dir = repo_dir / "features" / "feature_cache"

    with open(splits_dir / f"oasis1_primary_seed{seed}.json", "r") as f:
        oasis_split = json.load(f)
    with open(splits_dir / f"spanish_trackB_seed{seed}.json", "r") as f:
        sp_split = json.load(f)

    test_df = df_oasis[df_oasis["subject_id"].isin(set(oasis_split["test"]))].copy()
    y_test = test_df["label"].to_numpy(dtype=int)
    X_test_raw = test_df[FEATURE_ORDER].to_numpy(dtype=float)

    eval_df = df_sp[df_sp["subject_id"].isin(set(sp_split["evaluation"]))].copy()
    y_eval = eval_df["label"].to_numpy(dtype=int)
    X_eval_raw = eval_df.rename(columns=SPANISH_MAP)[FEATURE_ORDER].to_numpy(dtype=float)

    scaler = load_scaler(scalers_dir / f"scaler_seed{seed}.pkl")
    X_test_scaled = apply_scaler(scaler, X_test_raw)
    X_eval_scaled = apply_scaler(scaler, X_eval_raw)

    # Models
    lr_model = joblib.load(models_dir / f"lr_seed{seed}.pkl")
    svm_model = joblib.load(models_dir / f"svm_seed{seed}.pkl")

    mlp_model = MLPClassifier(input_dim=4, hidden_dims=(32, 16))
    mlp_model.load_state_dict(torch.load(models_dir / f"mlp_seed{seed}.pt", weights_only=True))
    mlp_model.eval()

    combat_mlp = MLPClassifier(input_dim=4, hidden_dims=(32, 16))
    combat_mlp.load_state_dict(torch.load(models_dir / f"combat_mlp_seed{seed}.pt", weights_only=True))
    combat_mlp.eval()

    dann_model = DANNArchitecture(input_dim=4)
    dann_model.load_state_dict(torch.load(models_dir / f"dann_seed{seed}.pt", weights_only=True))
    dann_model.eval()

    # Probabilities on OASIS test
    p_test_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    p_test_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
    with torch.no_grad():
        p_test_mlp = mlp_model(torch.tensor(X_test_scaled, dtype=torch.float32)).cpu().numpy().flatten()
        p_test_combat = combat_mlp(torch.tensor(X_test_scaled, dtype=torch.float32)).cpu().numpy().flatten()
        p_test_dann = dann_model.predict_label(torch.tensor(X_test_scaled, dtype=torch.float32)).cpu().numpy().flatten()

    # Probabilities on Spanish Eval
    p_eval_lr = lr_model.predict_proba(X_eval_scaled)[:, 1]
    p_eval_svm = svm_model.predict_proba(X_eval_scaled)[:, 1]
    with torch.no_grad():
        p_eval_mlp = mlp_model(torch.tensor(X_eval_scaled, dtype=torch.float32)).cpu().numpy().flatten()
        p_eval_combat = combat_mlp(torch.tensor(X_eval_scaled, dtype=torch.float32)).cpu().numpy().flatten()
        p_eval_dann = dann_model.predict_label(torch.tensor(X_eval_scaled, dtype=torch.float32)).cpu().numpy().flatten()

    # --- Plot 1: Internal ROC Curves ---
    plt.figure(figsize=(7, 6))
    for name, p in [
        ("LR", p_test_lr),
        ("SVM", p_test_svm),
        ("MLP", p_test_mlp),
        ("ComBat+MLP", p_test_combat),
        ("DANN", p_test_dann),
    ]:
        fpr, tpr, _ = roc_curve(y_test, p)
        auc = roc_auc_score(y_test, p)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("ROC Curves: OASIS-1 Held-Out Test Set", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_internal_all_methods.png", dpi=300)
    plt.close()

    # --- Plot 2: External ROC Curves ---
    plt.figure(figsize=(7, 6))
    for name, p in [
        ("LR", p_eval_lr),
        ("SVM", p_eval_svm),
        ("MLP", p_eval_mlp),
        ("ComBat+MLP", p_eval_combat),
        ("DANN", p_eval_dann),
    ]:
        fpr, tpr, _ = roc_curve(y_eval, p)
        auc = roc_auc_score(y_eval, p)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("ROC Curves: Spanish Cohort Evaluation", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_external_all_methods.png", dpi=300)
    plt.close()

    # --- Plot 3: Grouped Bar Chart (AUROC by Method) ---
    methods = list(summary_data.keys())
    oasis_means = [summary_data[m]["oasis_mean"] for m in methods]
    oasis_stds = [summary_data[m]["oasis_std"] for m in methods]
    sp_means = [summary_data[m]["sp_mean"] for m in methods]
    sp_stds = [summary_data[m]["sp_std"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(9, 5.5))
    plt.bar(x - width / 2, oasis_means, width, yerr=oasis_stds, label="OASIS-1 Test", capsize=5, color="#1f77b4")
    plt.bar(x + width / 2, sp_means, width, yerr=sp_stds, label="Spanish Cohort", capsize=5, color="#ff7f0e")
    plt.ylabel("AUROC", fontsize=11)
    plt.title("Internal vs External AUROC Across Harmonization Methods", fontsize=13, fontweight="bold")
    plt.xticks(x, methods, rotation=15, ha="right", fontsize=10)
    plt.ylim([0.4, 1.0])
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(plots_dir / "auroc_comparison_bar.png", dpi=300)
    plt.close()

    # --- Plot 4: Generalization Gap Bar Chart ---
    gaps = [summary_data[m]["gap_mean"] for m in methods]
    gap_stds = [summary_data[m]["gap_std"] for m in methods]

    plt.figure(figsize=(8, 5))
    colors = ["#d62728" if g > 0.1 else "#2ca02c" for g in gaps]
    plt.bar(methods, gaps, yerr=gap_stds, capsize=5, color=colors, alpha=0.85)
    plt.ylabel("Generalization Gap (OASIS - Spanish AUROC)", fontsize=11)
    plt.title("Cross-Scanner Generalization Gap by Method", fontsize=13, fontweight="bold")
    plt.axhline(0.0, color="black", lw=1)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "generalization_gap_bar.png", dpi=300)
    plt.close()
    print("All 4 plots generated successfully in results/plots/.")


def generate_final_benchmark_summary(
    results_dir: Path,
    summary_data: Dict[str, Dict[str, Any]],
) -> None:
    """Write results/final_comparison_table.csv and results/summary_milestone3.txt."""
    # 1. Save CSV
    records = []
    base_mlp_sp = summary_data["MLP (baseline)"]["sp_mean"]

    for m, d in summary_data.items():
        delta = (d["sp_mean"] - base_mlp_sp) if m in ["ComBat+MLP", "DANN"] else 0.0
        records.append({
            "Method": m,
            "OASIS_AUROC_mean": d["oasis_mean"],
            "OASIS_AUROC_std": d["oasis_std"],
            "Spanish_AUROC_mean": d["sp_mean"],
            "Spanish_AUROC_std": d["sp_std"],
            "Gap_mean": d["gap_mean"],
            "Gap_std": d["gap_std"],
            "Delta_vs_MLP": delta,
        })
    df_table = pd.DataFrame(records)
    csv_path = results_dir / "final_comparison_table.csv"
    df_table.to_csv(csv_path, index=False)
    print(f"Saved final comparison table to {csv_path}")

    # Determine best method on Spanish cohort
    best_method = max(summary_data.keys(), key=lambda m: summary_data[m]["sp_mean"])
    combat_rec = summary_data["ComBat+MLP"]["sp_mean"] - base_mlp_sp
    dann_rec = summary_data["DANN"]["sp_mean"] - base_mlp_sp

    # 2. Save summary_milestone3.txt
    summary_lines = [
        "=== MILESTONE 3 RESULTS ===",
        "=== FULL BENCHMARK COMPARISON ===",
        "",
        f"{'Method':<16} | {'OASIS AUROC':<15} | {'Spanish AUROC':<15} | {'Gap':<8}",
        f"{'-'*16}|{'-'*15}|{'-'*15}|{'-'*8}",
    ]

    for m, d in summary_data.items():
        summary_lines.append(
            f"{m:<16} | {d['oasis_mean']:.2f}±{d['oasis_std']:.2f}         | "
            f"{d['sp_mean']:.2f}±{d['sp_std']:.2f}         | {d['gap_mean']:+.2f}"
        )

    summary_lines.extend([
        "",
        f"Best method on Spanish cohort: {best_method}",
        f"Gap recovery by ComBat: {combat_rec:+.2f} AUROC points",
        f"Gap recovery by DANN:   {dann_rec:+.2f} AUROC points",
        "",
        "LIMITATIONS:",
        "- Spanish cohort n=54 (eval n≈43), CIs are wide",
        "- Cannot definitively rank ComBat vs DANN at this sample size",
        "- OASIS-1 has single scanner (no source domain diversity for DANN)",
        "- Feature pipelines differ between cohorts (FreeSurfer vs SPM)",
        "",
    ])

    summary_text = "\n".join(summary_lines)
    txt_path = results_dir / "summary_milestone3.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Saved summary to {txt_path}\n")
    print(summary_text)


def run_final_comparison():
    """Main execution entry point for final comparison and reporting."""
    repo_dir = Path(__file__).resolve().parent.parent
    results_dir = repo_dir / "results"
    plots_dir = results_dir / "plots"

    seeds = [42, 123, 456]
    summary_data = load_all_benchmark_results(results_dir, seeds)
    generate_comparison_plots(repo_dir, plots_dir, summary_data, seed=42)
    generate_final_benchmark_summary(results_dir, summary_data)


if __name__ == "__main__":
    run_final_comparison()
