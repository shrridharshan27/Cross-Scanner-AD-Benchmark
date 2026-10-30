"""Internal Evaluation Pipeline on OASIS-1 Held-Out Test Set.

Evaluates trained baseline models (LR, SVM, MLP) on the held-out OASIS-1 test set
for seeds [42, 123, 456]. Classification thresholds are selected on the validation set
using Youden's J statistic and frozen for test evaluation. Bootstrap 95% confidence
intervals are computed for AUROC.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch

from ad_generalization_benchmark.configs import load_experiment_config
from ad_generalization_benchmark.data.loaders.oasis1_loader import load_oasis1
from ad_generalization_benchmark.features.align_features import apply_scaler, load_scaler
from ad_generalization_benchmark.models.mlp import MLPClassifier

FEATURE_ORDER = ["nWBV", "eTIV", "age", "sex"]


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find classification threshold maximizing Youden's J index (sensitivity + specificity - 1).

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (1).

    Returns:
        Optimal threshold bounded in [0.0, 1.0].
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr

    # Filter out thresholds that are infinite or out of probability bounds
    valid_indices = [i for i, t in enumerate(thresholds) if 0.0 <= t <= 1.0]
    if not valid_indices:
        return 0.5

    best_idx = valid_indices[int(np.argmax(j_scores[valid_indices]))]
    return float(thresholds[best_idx])


def compute_bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute non-parametric bootstrap (percentile) 95% confidence interval for AUROC.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted positive-class probabilities.
        n_bootstraps: Number of bootstrap resamples (default: 1000).
        alpha: Significance level (default: 0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (ci_low, ci_high): 95% bootstrap confidence interval bounds.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    boot_aurocs = []

    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n_samples, size=n_samples)
        y_boot = y_true[boot_idx]
        prob_boot = y_prob[boot_idx]

        # AUROC requires at least one positive and one negative sample
        if len(np.unique(y_boot)) < 2:
            continue
        boot_aurocs.append(roc_auc_score(y_boot, prob_boot))

    if not boot_aurocs:
        base_score = float(roc_auc_score(y_true, y_prob))
        return base_score, base_score

    ci_low = float(np.percentile(boot_aurocs, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_aurocs, 100.0 * (1.0 - alpha / 2.0)))
    return ci_low, ci_high


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    seed: int = 42,
    n_bootstraps: int = 1000,
) -> Dict[str, Any]:
    """Compute AUROC, 95% CI, balanced accuracy, sensitivity, specificity, and confusion matrix."""
    y_pred = (y_prob >= threshold).astype(int)
    auroc = float(roc_auc_score(y_true, y_prob))
    ci_low, ci_high = compute_bootstrap_auroc_ci(y_true, y_prob, n_bootstraps=n_bootstraps, seed=seed)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    sens = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    spec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    conf_mat = confusion_matrix(y_true, y_pred).tolist()

    return {
        "auroc": auroc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "threshold": float(threshold),
        "balanced_accuracy": bal_acc,
        "sensitivity": sens,
        "specificity": spec,
        "confusion_matrix": conf_mat,
    }


def evaluate_single_seed(
    seed: int,
    df_oasis: pd.DataFrame,
    splits_dir: Path,
    scalers_dir: Path,
    models_dir: Path,
    results_dir: Path,
) -> Dict[str, Any]:
    """Evaluate LR, SVM, and MLP on held-out OASIS-1 test set for a single seed."""
    split_path = splits_dir / f"oasis1_primary_seed{seed}.json"
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    val_df = df_oasis[df_oasis["subject_id"].isin(set(split_data["val"]))].copy()
    test_df = df_oasis[df_oasis["subject_id"].isin(set(split_data["test"]))].copy()

    X_val_raw = val_df[FEATURE_ORDER].to_numpy(dtype=float)
    y_val = val_df["label"].to_numpy(dtype=int)

    X_test_raw = test_df[FEATURE_ORDER].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy(dtype=int)

    # Load fitted scaler (fitted strictly on train partition)
    scaler_path = scalers_dir / f"scaler_seed{seed}.pkl"
    scaler = load_scaler(scaler_path)

    # Scale continuous features using frozen train scaler
    X_val_scaled = apply_scaler(scaler, X_val_raw)
    X_test_scaled = apply_scaler(scaler, X_test_raw)

    # Load models
    lr_model = joblib.load(models_dir / f"lr_seed{seed}.pkl")
    svm_model = joblib.load(models_dir / f"svm_seed{seed}.pkl")

    mlp_model = MLPClassifier(input_dim=4, hidden_dims=(32, 16))
    mlp_state = torch.load(models_dir / f"mlp_seed{seed}.pt", weights_only=True)
    mlp_model.load_state_dict(mlp_state)
    mlp_model.eval()

    # Val probabilities
    val_probs_lr = lr_model.predict_proba(X_val_scaled)[:, 1]
    val_probs_svm = svm_model.predict_proba(X_val_scaled)[:, 1]
    with torch.no_grad():
        val_probs_mlp = mlp_model(torch.tensor(X_val_scaled, dtype=torch.float32)).cpu().numpy().flatten()

    # Test probabilities
    test_probs_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    test_probs_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
    with torch.no_grad():
        test_probs_mlp = mlp_model(torch.tensor(X_test_scaled, dtype=torch.float32)).cpu().numpy().flatten()

    # Select threshold on validation set via Youden's J
    thresh_lr = find_optimal_threshold(y_val, val_probs_lr)
    thresh_svm = find_optimal_threshold(y_val, val_probs_svm)
    thresh_mlp = find_optimal_threshold(y_val, val_probs_mlp)

    # Evaluate on test set with frozen threshold
    res_lr = compute_classification_metrics(y_test, test_probs_lr, thresh_lr, seed=seed)
    res_lr["val_auroc"] = float(roc_auc_score(y_val, val_probs_lr))

    res_svm = compute_classification_metrics(y_test, test_probs_svm, thresh_svm, seed=seed)
    res_svm["val_auroc"] = float(roc_auc_score(y_val, val_probs_svm))

    res_mlp = compute_classification_metrics(y_test, test_probs_mlp, thresh_mlp, seed=seed)
    res_mlp["val_auroc"] = float(roc_auc_score(y_val, val_probs_mlp))

    seed_result = {
        "seed": seed,
        "test_n": len(y_test),
        "test_n_pos": int((y_test == 1).sum()),
        "test_n_neg": int((y_test == 0).sum()),
        "lr": res_lr,
        "svm": res_svm,
        "mlp": res_mlp,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"internal_eval_seed{seed}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(seed_result, f, indent=2)

    return seed_result


def evaluate_internal_all() -> Dict[int, Any]:
    """Run internal evaluation across all seeds and print aggregated summary table."""
    exp_cfg = load_experiment_config()
    seeds = exp_cfg.get("seeds", [42, 123, 456])

    repo_dir = Path(__file__).resolve().parent.parent
    splits_dir = repo_dir / "splits" / "saved"
    scalers_dir = repo_dir / "features" / "feature_cache"
    models_dir = repo_dir / "models" / "saved"
    results_dir = repo_dir / "results"

    df_oasis, _ = load_oasis1(verbose=False)
    all_results = {}

    print("=" * 70)
    print("INTERNAL EVALUATION (OASIS-1 HELD-OUT TEST SET)")
    print("=" * 70)

    for s in seeds:
        res = evaluate_single_seed(s, df_oasis, splits_dir, scalers_dir, models_dir, results_dir)
        all_results[s] = res
        print(f"Seed {s:3d} saved to: results/internal_eval_seed{s}.json")

    # Aggregate metrics across seeds
    models = ["lr", "svm", "mlp"]
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: OASIS-1 HELD-OUT TEST SET (Mean ± Std over 3 seeds)")
    print("=" * 70)
    print(f"{'Model':<8} | {'AUROC':<16} | {'BalAcc':<16} | {'Sensitivity':<16} | {'Specificity':<16}")
    print("-" * 70)

    for m in models:
        aurocs = [all_results[s][m]["auroc"] for s in seeds]
        bal_accs = [all_results[s][m]["balanced_accuracy"] for s in seeds]
        senss = [all_results[s][m]["sensitivity"] for s in seeds]
        specs = [all_results[s][m]["specificity"] for s in seeds]

        name = m.upper()
        print(
            f"{name:<8} | "
            f"{np.mean(aurocs):.4f} ± {np.std(aurocs):.4f} | "
            f"{np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f} | "
            f"{np.mean(senss):.4f} ± {np.std(senss):.4f} | "
            f"{np.mean(specs):.4f} ± {np.std(specs):.4f}"
        )
    print("=" * 70)
    return all_results


if __name__ == "__main__":
    evaluate_internal_all()
