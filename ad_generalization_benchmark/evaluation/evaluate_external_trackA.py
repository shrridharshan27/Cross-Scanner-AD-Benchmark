"""External Evaluation Pipeline — Track A (Zero Adaptation Baseline).

Evaluates models trained strictly on OASIS-1 against the Spanish MRI cohort (N=54).
The Spanish cohort is never used in any fitting, scaling, or threshold-selection step.
Continuous features are scaled using the frozen OASIS-1 training scaler, and classification
thresholds are frozen from the OASIS-1 validation set (Youden's J).
Computes signed generalization gaps and 95% bootstrap confidence intervals.
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
)
import torch

from ad_generalization_benchmark.configs import load_experiment_config
from ad_generalization_benchmark.evaluation.evaluate import compute_bootstrap_auroc_ci
from ad_generalization_benchmark.features.align_features import apply_scaler, load_scaler
from ad_generalization_benchmark.models.mlp import MLPClassifier

SPANISH_FEATURE_MAP = {
    "brain_vol_ratio": "nWBV",
    "icv_estimate": "eTIV",
    "age": "age",
    "sex": "sex",
}
FEATURE_ORDER = ["nWBV", "eTIV", "age", "sex"]


def load_spanish_dataset(cache_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and format Spanish feature cache.

    Args:
        cache_path: Path to cached spanish_features.csv.

    Returns:
        (X, y, subject_ids)
    """
    if not cache_path.exists():
        raise FileNotFoundError(f"Spanish feature cache not found at: {cache_path}")

    df = pd.read_csv(cache_path)
    # Check that required columns exist
    for col in ["brain_vol_ratio", "icv_estimate", "age", "sex", "label"]:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' missing from Spanish features.")

    renamed = df.rename(columns=SPANISH_FEATURE_MAP)
    X = renamed[FEATURE_ORDER].to_numpy(dtype=float)
    y = renamed["label"].to_numpy(dtype=int)
    subject_ids = df["subject_id"].tolist()
    return X, y, subject_ids


def evaluate_external_for_seed(
    seed: int,
    X_spanish_raw: np.ndarray,
    y_spanish: np.ndarray,
    scalers_dir: Path,
    models_dir: Path,
    results_dir: Path,
) -> Dict[str, Any]:
    """Perform Track A external evaluation on Spanish cohort for a single seed."""
    # 1. Load internal evaluation result to retrieve frozen threshold & OASIS test AUROC
    internal_res_file = results_dir / f"internal_eval_seed{seed}.json"
    if not internal_res_file.exists():
        raise FileNotFoundError(
            f"Internal evaluation results not found at {internal_res_file}. "
            f"Run evaluate.py before running external evaluation."
        )
    with open(internal_res_file, "r", encoding="utf-8") as f:
        internal_data = json.load(f)

    # 2. Load frozen scaler fitted on OASIS-1 train partition
    scaler_path = scalers_dir / f"scaler_seed{seed}.pkl"
    scaler = load_scaler(scaler_path)

    # Apply scaler to Spanish continuous features
    X_spanish_scaled = apply_scaler(scaler, X_spanish_raw)

    # 3. Load models
    lr_model = joblib.load(models_dir / f"lr_seed{seed}.pkl")
    svm_model = joblib.load(models_dir / f"svm_seed{seed}.pkl")

    mlp_model = MLPClassifier(input_dim=4, hidden_dims=(32, 16))
    mlp_state = torch.load(models_dir / f"mlp_seed{seed}.pt", weights_only=True)
    mlp_model.load_state_dict(mlp_state)
    mlp_model.eval()

    # 4. Predict probabilities
    probs_lr = lr_model.predict_proba(X_spanish_scaled)[:, 1]
    probs_svm = svm_model.predict_proba(X_spanish_scaled)[:, 1]
    with torch.no_grad():
        probs_mlp = mlp_model(torch.tensor(X_spanish_scaled, dtype=torch.float32)).cpu().numpy().flatten()

    models_eval = {}
    model_preds = {
        "lr": probs_lr,
        "svm": probs_svm,
        "mlp": probs_mlp,
    }

    for model_name, probs in model_preds.items():
        frozen_thresh = internal_data[model_name]["threshold"]
        oasis_test_auroc = internal_data[model_name]["auroc"]

        y_pred = (probs >= frozen_thresh).astype(int)
        sp_auroc = float(roc_auc_score(y_spanish, probs))
        ci_low, ci_high = compute_bootstrap_auroc_ci(y_spanish, probs, n_bootstraps=1000, seed=seed)
        bal_acc = float(balanced_accuracy_score(y_spanish, y_pred))
        sens = float(recall_score(y_spanish, y_pred, pos_label=1, zero_division=0))
        spec = float(recall_score(y_spanish, y_pred, pos_label=0, zero_division=0))
        conf_mat = confusion_matrix(y_spanish, y_pred).tolist()

        # Signed generalization gap: oasis_test_auroc - spanish_auroc
        gap_auroc = float(oasis_test_auroc - sp_auroc)

        models_eval[model_name] = {
            "oasis_test_auroc": oasis_test_auroc,
            "spanish_auroc": sp_auroc,
            "gap_auroc": gap_auroc,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "frozen_threshold": frozen_thresh,
            "balanced_accuracy": bal_acc,
            "sensitivity": sens,
            "specificity": spec,
            "confusion_matrix": conf_mat,
        }

    seed_result = {
        "seed": seed,
        "spanish_n": len(y_spanish),
        "spanish_n_pos": int((y_spanish == 1).sum()),
        "spanish_n_neg": int((y_spanish == 0).sum()),
        "warning": "Spanish n=54, CIs will be wide — do not rank methods definitively",
        "lr": models_eval["lr"],
        "svm": models_eval["svm"],
        "mlp": models_eval["mlp"],
    }

    out_file = results_dir / f"external_trackA_seed{seed}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(seed_result, f, indent=2)

    return seed_result


def evaluate_external_all() -> Dict[int, Any]:
    """Run Track A external evaluation across all seeds and print summary table."""
    exp_cfg = load_experiment_config()
    seeds = exp_cfg.get("seeds", [42, 123, 456])

    repo_dir = Path(__file__).resolve().parent.parent
    features_cache = repo_dir / "features" / "feature_cache" / "spanish_features.csv"
    scalers_dir = repo_dir / "features" / "feature_cache"
    models_dir = repo_dir / "models" / "saved"
    results_dir = repo_dir / "results"

    print("=" * 80)
    print("EXTERNAL EVALUATION — TRACK A (Zero Adaptation Baseline)")
    print("=" * 80)
    print("NOTE / WARNING: Spanish n=54, CIs will be wide — do not rank methods definitively\n")

    X_sp, y_sp, ids = load_spanish_dataset(features_cache)
    print(f"Loaded Spanish cohort: {len(X_sp)} subjects (AD={int((y_sp==1).sum())}, CN={int((y_sp==0).sum())})")

    all_results = {}
    for s in seeds:
        res = evaluate_external_for_seed(s, X_sp, y_sp, scalers_dir, models_dir, results_dir)
        all_results[s] = res
        print(f"Seed {s:3d} saved to: results/external_trackA_seed{s}.json")

    # Aggregate and print summary table
    models = ["lr", "svm", "mlp"]
    print("\n" + "=" * 80)
    print(f"{'Model':<8} | {'OASIS AUROC':<16} | {'Spanish AUROC':<16} | {'Gap (OASIS-SP)':<16} | {'Spanish 95% CI':<16}")
    print("-" * 80)

    for m in models:
        oasis_aurocs = [all_results[s][m]["oasis_test_auroc"] for s in seeds]
        sp_aurocs = [all_results[s][m]["spanish_auroc"] for s in seeds]
        gaps = [all_results[s][m]["gap_auroc"] for s in seeds]
        ci_lows = [all_results[s][m]["ci_low"] for s in seeds]
        ci_highs = [all_results[s][m]["ci_high"] for s in seeds]

        name = m.upper()
        print(
            f"{name:<8} | "
            f"{np.mean(oasis_aurocs):.4f} ± {np.std(oasis_aurocs):.4f} | "
            f"{np.mean(sp_aurocs):.4f} ± {np.std(sp_aurocs):.4f} | "
            f"{np.mean(gaps):+.4f} ± {np.std(gaps):.4f} | "
            f"[{np.mean(ci_lows):.4f}, {np.mean(ci_highs):.4f}]"
        )
    print("=" * 80)
    return all_results


if __name__ == "__main__":
    evaluate_external_all()
