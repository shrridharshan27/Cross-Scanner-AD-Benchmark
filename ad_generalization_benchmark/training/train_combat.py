"""Track B: ComBat Feature-Level Harmonization Pipeline.

ComBat removes scanner and site batch effects from neuroimaging features while preserving
biological covariates (age, sex). ComBat models each feature across subjects i and batches j as:
    y_ij = alpha + X * beta + gamma_j + delta_j * epsilon_ij
where gamma_j and delta_j represent the additive and multiplicative batch effects of site j.
Biological covariates (age, sex) are included in the design matrix X so that legitimate biological
variation is NOT removed. Diagnosis labels and CDR scores are NEVER included in ComBat fitting.

Strict Leakage Controls:
- ComBat is fitted ONLY on OASIS-1 train features + Spanish calibration features (~11 unlabeled subjects).
- Spanish evaluation subjects (~43 subjects) are NEVER used during ComBat fitting; they are transformed
  out-of-sample using the fitted ComBat parameters (estimates).
- Scaler is fitted strictly on the ComBat-harmonized OASIS-1 training features.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from neuroCombat import neuroCombat, neuroCombatFromTraining
except ImportError:
    raise ImportError("neuroCombat is required for Track B. Run: pip install neuroCombat")

from ad_generalization_benchmark.configs import load_experiment_config
from ad_generalization_benchmark.data.loaders.oasis1_loader import load_oasis1
from ad_generalization_benchmark.evaluation.evaluate import (
    compute_bootstrap_auroc_ci,
    find_optimal_threshold,
)
from ad_generalization_benchmark.features.align_features import fit_and_apply_scaler, apply_scaler
from ad_generalization_benchmark.models.mlp import MLPClassifier

FEATURE_ORDER = ["nWBV", "eTIV", "age", "sex"]
SPANISH_MAP = {
    "brain_vol_ratio": "nWBV",
    "icv_estimate": "eTIV",
    "age": "age",
    "sex": "sex",
}


def harmonize_dataset_with_combat(
    oasis_train_df: pd.DataFrame,
    oasis_val_df: pd.DataFrame,
    oasis_test_df: pd.DataFrame,
    spanish_calib_df: pd.DataFrame,
    spanish_eval_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Harmonize features across OASIS-1 and Spanish cohorts using neuroCombat.

    Fitting Data: OASIS-1 train + Spanish calibration (unlabeled, ~11 subjects).
    Out-of-sample Transformation: OASIS val, OASIS test, and Spanish eval sets.

    Returns:
        oasis_train_harm: shape (N_oasis_train, 4)
        oasis_val_harm:   shape (N_oasis_val, 4)
        oasis_test_harm:  shape (N_oasis_test, 4)
        spanish_eval_harm: shape (N_sp_eval, 4)
        combat_estimates: Dictionary of fitted ComBat parameters
    """
    # 1. Prepare raw features (transposed: features x subjects for neuroCombat)
    oasis_tr_X = oasis_train_df[FEATURE_ORDER].to_numpy(dtype=float)
    sp_calib_X = spanish_calib_df.rename(columns=SPANISH_MAP)[FEATURE_ORDER].to_numpy(dtype=float)

    n_oasis_tr = len(oasis_tr_X)
    n_sp_calib = len(sp_calib_X)

    # ComBat requires >= 2 subjects per batch
    # Spanish calibration set is small (~11 subjects). This may cause numerical instability.
    if n_sp_calib < 2:
        raise ValueError(f"ComBat requires at least 2 subjects in calibration batch, got {n_sp_calib}")

    combined_X = np.vstack([oasis_tr_X, sp_calib_X])  # subjects x features

    covars = pd.DataFrame({
        "batch": ["oasis"] * n_oasis_tr + ["spanish"] * n_sp_calib,
        "age": np.concatenate([oasis_train_df["age"].values, spanish_calib_df["age"].values]),
        "sex": np.concatenate([oasis_train_df["sex"].values, spanish_calib_df["sex"].values]),
    })

    # Fit ComBat on OASIS train + Spanish calibration only
    # Note: neuroCombat takes dat shape (features, subjects)
    combat_fit = neuroCombat(
        dat=combined_X.T,
        covars=covars,
        batch_col="batch",
        continuous_cols=["age"],
        categorical_cols=["sex"],
        parametric=True,
    )

    harmonized_data = combat_fit["data"].T  # subjects x features
    oasis_train_harm = harmonized_data[:n_oasis_tr]
    estimates = combat_fit["estimates"]

    # Out-of-sample harmonization on OASIS validation set
    oasis_val_X = oasis_val_df[FEATURE_ORDER].to_numpy(dtype=float)
    val_out = neuroCombatFromTraining(
        dat=oasis_val_X.T,
        batch=["oasis"] * len(oasis_val_X),
        estimates=estimates,
    )
    oasis_val_harm = val_out["data"].T

    # Out-of-sample harmonization on OASIS held-out test set
    oasis_test_X = oasis_test_df[FEATURE_ORDER].to_numpy(dtype=float)
    test_out = neuroCombatFromTraining(
        dat=oasis_test_X.T,
        batch=["oasis"] * len(oasis_test_X),
        estimates=estimates,
    )
    oasis_test_harm = test_out["data"].T

    # Out-of-sample harmonization on Spanish evaluation set (unlabeled during fit)
    sp_eval_X = spanish_eval_df.rename(columns=SPANISH_MAP)[FEATURE_ORDER].to_numpy(dtype=float)
    sp_eval_out = neuroCombatFromTraining(
        dat=sp_eval_X.T,
        batch=["spanish"] * len(sp_eval_X),
        estimates=estimates,
    )
    spanish_eval_harm = sp_eval_out["data"].T

    return oasis_train_harm, oasis_val_harm, oasis_test_harm, spanish_eval_harm, estimates


def train_combat_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    lr: float = 1e-3,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 16,
) -> Tuple[MLPClassifier, float, float, Dict[str, Any]]:
    """Train 3-layer MLP on ComBat-harmonized features with identical baseline architecture."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    model = MLPClassifier(input_dim=4, hidden_dims=(32, 16))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_tensor_x = torch.tensor(X_train, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(train_tensor_x, train_tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_tensor_x = torch.tensor(X_val, dtype=torch.float32)

    best_val_auroc = -1.0
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            weights = torch.where(batch_y == 1.0, torch.tensor(pos_weight), torch.tensor(1.0))
            loss = -(weights * (batch_y * torch.log(preds + 1e-7) + (1.0 - batch_y) * torch.log(1.0 - preds + 1e-7))).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = model(val_tensor_x).cpu().numpy().flatten()
            val_auroc = float(roc_auc_score(y_val, val_probs))

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_probs = model(train_tensor_x).cpu().numpy().flatten()
        train_auroc = float(roc_auc_score(y_train, train_probs))

    hp_info = {"lr": lr, "best_epoch": best_epoch, "pos_weight": pos_weight}
    return model, train_auroc, best_val_auroc, hp_info


def run_combat_for_seed(
    seed: int,
    df_oasis: pd.DataFrame,
    df_spanish: pd.DataFrame,
    splits_dir: Path,
    models_dir: Path,
    results_dir: Path,
) -> Dict[str, Any]:
    """Execute ComBat harmonization, model training, and evaluation for a single seed."""
    # 1. Load splits
    oasis_split_file = splits_dir / f"oasis1_primary_seed{seed}.json"
    sp_split_file = splits_dir / f"spanish_trackB_seed{seed}.json"

    with open(oasis_split_file, "r", encoding="utf-8") as f:
        oasis_split = json.load(f)
    with open(sp_split_file, "r", encoding="utf-8") as f:
        sp_split = json.load(f)

    oasis_train_df = df_oasis[df_oasis["subject_id"].isin(set(oasis_split["train"]))].copy()
    oasis_val_df = df_oasis[df_oasis["subject_id"].isin(set(oasis_split["val"]))].copy()
    oasis_test_df = df_oasis[df_oasis["subject_id"].isin(set(oasis_split["test"]))].copy()

    sp_calib_df = df_spanish[df_spanish["subject_id"].isin(set(sp_split["calibration"]))].copy()
    sp_eval_df = df_spanish[df_spanish["subject_id"].isin(set(sp_split["evaluation"]))].copy()

    # Verify no eval subject in calibration
    assert len(set(sp_split["calibration"]) & set(sp_split["evaluation"])) == 0

    y_train = oasis_train_df["label"].to_numpy(dtype=int)
    y_val = oasis_val_df["label"].to_numpy(dtype=int)
    y_test = oasis_test_df["label"].to_numpy(dtype=int)
    y_sp_eval = sp_eval_df["label"].to_numpy(dtype=int)

    # 2. Run ComBat
    try:
        tr_harm, val_harm, te_harm, sp_eval_harm, _ = harmonize_dataset_with_combat(
            oasis_train_df, oasis_val_df, oasis_test_df, sp_calib_df, sp_eval_df
        )
    except Exception as e:
        print(f"[WARNING] ComBat failed on Seed {seed}: {e}. Skipping this seed.")
        return {"seed": seed, "status": "failed", "error": str(e)}

    # 3. Fit StandardScaler strictly on ComBat-harmonized OASIS train continuous features
    scaler, [tr_scaled, val_scaled, te_scaled, sp_eval_scaled] = fit_and_apply_scaler(
        tr_harm, val_harm, te_harm, sp_eval_harm
    )

    # 4. Train MLP on harmonized scaled features
    mlp_model, tr_auc, val_auc, hp_info = train_combat_mlp(
        tr_scaled, y_train, val_scaled, y_val, seed=seed
    )

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"combat_mlp_seed{seed}.pt"
    torch.save(mlp_model.state_dict(), model_path)

    # 5. Evaluate on OASIS val (for threshold) and test
    with torch.no_grad():
        val_probs = mlp_model(torch.tensor(val_scaled, dtype=torch.float32)).cpu().numpy().flatten()
        test_probs = mlp_model(torch.tensor(te_scaled, dtype=torch.float32)).cpu().numpy().flatten()
        sp_eval_probs = mlp_model(torch.tensor(sp_eval_scaled, dtype=torch.float32)).cpu().numpy().flatten()

    # Optimal threshold from validation set
    threshold = find_optimal_threshold(y_val, val_probs)

    # Internal OASIS test metrics
    test_preds = (test_probs >= threshold).astype(int)
    internal_auroc = float(roc_auc_score(y_test, test_probs))
    int_ci_low, int_ci_high = compute_bootstrap_auroc_ci(y_test, test_probs, n_bootstraps=1000, seed=seed)
    int_bal_acc = float(balanced_accuracy_score(y_test, test_preds))
    int_sens = float(recall_score(y_test, test_preds, pos_label=1, zero_division=0))
    int_spec = float(recall_score(y_test, test_preds, pos_label=0, zero_division=0))

    # External Spanish eval metrics
    sp_preds = (sp_eval_probs >= threshold).astype(int)
    external_auroc = float(roc_auc_score(y_sp_eval, sp_eval_probs))
    ext_ci_low, ext_ci_high = compute_bootstrap_auroc_ci(y_sp_eval, sp_eval_probs, n_bootstraps=1000, seed=seed)
    ext_bal_acc = float(balanced_accuracy_score(y_sp_eval, sp_preds))
    ext_sens = float(recall_score(y_sp_eval, sp_preds, pos_label=1, zero_division=0))
    ext_spec = float(recall_score(y_sp_eval, sp_preds, pos_label=0, zero_division=0))

    # Generalization gap
    gap = internal_auroc - external_auroc

    # Load unadapted baseline MLP Spanish AUROC for comparison
    base_file = results_dir / f"external_trackA_seed{seed}.json"
    unadapted_sp_auroc = None
    if base_file.exists():
        with open(base_file, "r", encoding="utf-8") as bf:
            unadapted_data = json.load(bf)
            unadapted_sp_auroc = unadapted_data["mlp"]["spanish_auroc"]

    improvement = (external_auroc - unadapted_sp_auroc) if unadapted_sp_auroc is not None else None

    result = {
        "seed": seed,
        "method": "ComBat+MLP",
        "threshold": threshold,
        "internal": {
            "auroc": internal_auroc,
            "ci_low": int_ci_low,
            "ci_high": int_ci_high,
            "balanced_accuracy": int_bal_acc,
            "sensitivity": int_sens,
            "specificity": int_spec,
            "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
        },
        "external": {
            "spanish_eval_n": len(y_sp_eval),
            "auroc": external_auroc,
            "ci_low": ext_ci_low,
            "ci_high": ext_ci_high,
            "balanced_accuracy": ext_bal_acc,
            "sensitivity": ext_sens,
            "specificity": ext_spec,
            "confusion_matrix": confusion_matrix(y_sp_eval, sp_preds).tolist(),
        },
        "generalization_gap": gap,
        "unadapted_baseline_sp_auroc": unadapted_sp_auroc,
        "adaptation_improvement": improvement,
    }

    out_file = results_dir / f"combat_eval_seed{seed}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    imp_str = f"{improvement:+.4f}" if improvement is not None else "N/A"
    print(
        f"ComBat | Seed {seed:3d} | Internal AUROC: {internal_auroc:.4f} | "
        f"Spanish AUROC: {external_auroc:.4f} | Improvement: {imp_str}"
    )

    return result


def train_and_eval_combat_all() -> Dict[int, Any]:
    """Train and evaluate ComBat+MLP across all seeds [42, 123, 456]."""
    exp_cfg = load_experiment_config()
    seeds = exp_cfg.get("seeds", [42, 123, 456])

    repo_dir = Path(__file__).resolve().parent.parent
    splits_dir = repo_dir / "splits" / "saved"
    models_dir = repo_dir / "models" / "saved"
    results_dir = repo_dir / "results"
    sp_cache = repo_dir / "features" / "feature_cache" / "spanish_features.csv"

    print("=" * 75)
    print("TRACK B: ComBat HARMONIZATION + MLP BENCHMARK")
    print("=" * 75)
    print(f"Seeds: {seeds}")

    df_oasis, _ = load_oasis1(verbose=False)
    df_spanish = pd.read_csv(sp_cache)

    all_results = {}
    for s in seeds:
        res = run_combat_for_seed(s, df_oasis, df_spanish, splits_dir, models_dir, results_dir)
        all_results[s] = res

    # Summary table
    valid_res = [res for res in all_results.values() if res.get("status") != "failed"]
    if valid_res:
        int_aucs = [r["internal"]["auroc"] for r in valid_res]
        ext_aucs = [r["external"]["auroc"] for r in valid_res]
        gaps = [r["generalization_gap"] for r in valid_res]
        imps = [r["adaptation_improvement"] for r in valid_res if r["adaptation_improvement"] is not None]

        print("\n" + "=" * 75)
        print("ComBat SUMMARY (Mean ± Std over valid seeds)")
        print("=" * 75)
        print(f"Internal OASIS AUROC:  {np.mean(int_aucs):.4f} ± {np.std(int_aucs):.4f}")
        print(f"External Spanish AUROC: {np.mean(ext_aucs):.4f} ± {np.std(ext_aucs):.4f}")
        print(f"Generalization Gap:     {np.mean(gaps):+.4f} ± {np.std(gaps):.4f}")
        if imps:
            print(f"Adaptation Improvement: {np.mean(imps):+.4f} ± {np.std(imps):.4f} AUROC points")
        print("=" * 75)

    return all_results


if __name__ == "__main__":
    train_and_eval_combat_all()
