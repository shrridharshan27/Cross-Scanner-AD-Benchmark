"""Baseline Model Training Pipeline (Logistic Regression, Linear SVM, and MLP).

Trains three baseline models strictly on OASIS-1 training partition across
seeds [42, 123, 456]. Hyperparameters are tuned strictly on OASIS-1 validation
partitions, and fitted models and scalers are saved to disk.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ad_generalization_benchmark.configs import load_experiment_config
from ad_generalization_benchmark.data.loaders.oasis1_loader import load_oasis1
from ad_generalization_benchmark.features.align_features import fit_and_apply_scaler, save_scaler
from ad_generalization_benchmark.models.mlp import MLPClassifier

FEATURE_ORDER = ["nWBV", "eTIV", "age", "sex"]


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    C_grid: List[float] = [0.01, 0.1, 1.0, 10.0],
) -> Tuple[LogisticRegression, float, float, float]:
    """Train Logistic Regression with C selection on validation set."""
    best_c = C_grid[0]
    best_val_auroc = -1.0
    best_model = None

    for c in C_grid:
        model = LogisticRegression(
            C=c,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(X_train, y_train)

        val_probs = model.predict_proba(X_val)[:, 1]
        val_auroc = roc_auc_score(y_val, val_probs)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_c = c
            best_model = model

    train_probs = best_model.predict_proba(X_train)[:, 1]
    train_auroc = roc_auc_score(y_train, train_probs)

    return best_model, train_auroc, best_val_auroc, best_c


def train_linear_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    C_grid: List[float] = [0.01, 0.1, 1.0, 10.0],
) -> Tuple[CalibratedClassifierCV, float, float, float]:
    """Train Linear SVM with Platt scaling / calibration and C selection on validation set."""
    best_c = C_grid[0]
    best_val_auroc = -1.0
    best_calibrated = None

    for c in C_grid:
        base_svm = LinearSVC(
            C=c,
            class_weight="balanced",
            random_state=seed,
            max_iter=3000,
            dual="auto",
        )
        # Wrap with CalibratedClassifierCV to provide probability outputs
        calibrated = CalibratedClassifierCV(estimator=base_svm, cv=3)
        calibrated.fit(X_train, y_train)

        val_probs = calibrated.predict_proba(X_val)[:, 1]
        val_auroc = roc_auc_score(y_val, val_probs)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_c = c
            best_calibrated = calibrated

    train_probs = best_calibrated.predict_proba(X_train)[:, 1]
    train_auroc = roc_auc_score(y_train, train_probs)

    return best_calibrated, train_auroc, best_val_auroc, best_c


def train_mlp_pytorch(
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
    """Train 3-layer PyTorch MLP with early stopping on validation AUROC."""
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
            # Weighted binary cross entropy
            weights = torch.where(batch_y == 1.0, torch.tensor(pos_weight), torch.tensor(1.0))
            loss = -(weights * (batch_y * torch.log(preds + 1e-7) + (1.0 - batch_y) * torch.log(1.0 - preds + 1e-7))).mean()
            loss.backward()
            optimizer.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_probs = model(val_tensor_x).cpu().numpy()
            val_auroc = roc_auc_score(y_val, val_probs)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_probs = model(train_tensor_x).cpu().numpy()
        train_auroc = roc_auc_score(y_train, train_probs)

    hp_info = {"lr": lr, "best_epoch": best_epoch, "pos_weight": pos_weight}
    return model, train_auroc, best_val_auroc, hp_info


def run_training_for_seed(
    seed: int,
    df_oasis: pd.DataFrame,
    splits_dir: Path,
    scalers_dir: Path,
    models_dir: Path,
) -> Dict[str, Any]:
    """Execute complete baseline training pipeline for a single seed."""
    split_path = splits_dir / f"oasis1_primary_seed{seed}.json"
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_ids = set(split_data["train"])
    val_ids = set(split_data["val"])
    test_ids = set(split_data["test"])

    train_df = df_oasis[df_oasis["subject_id"].isin(train_ids)].copy()
    val_df = df_oasis[df_oasis["subject_id"].isin(val_ids)].copy()
    test_df = df_oasis[df_oasis["subject_id"].isin(test_ids)].copy()

    X_train_raw = train_df[FEATURE_ORDER].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)

    X_val_raw = val_df[FEATURE_ORDER].to_numpy(dtype=float)
    y_val = val_df["label"].to_numpy(dtype=int)

    X_test_raw = test_df[FEATURE_ORDER].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy(dtype=int)

    # Fit scaler strictly on training set (continuous features)
    scaler, [X_train, X_val, X_test] = fit_and_apply_scaler(X_train_raw, X_val_raw, X_test_raw)
    scaler_file = scalers_dir / f"scaler_seed{seed}.pkl"
    save_scaler(scaler, scaler_file)

    # 1. Logistic Regression
    lr_model, lr_tr_auc, lr_val_auc, lr_best_c = train_logistic_regression(
        X_train, y_train, X_val, y_val, seed=seed
    )
    lr_path = models_dir / f"lr_seed{seed}.pkl"
    joblib.dump(lr_model, lr_path)
    print(f"LR  | Seed {seed:3d} | Train AUROC: {lr_tr_auc:.4f} | Val AUROC: {lr_val_auc:.4f} | Best C: {lr_best_c}")

    # 2. Linear SVM
    svm_model, svm_tr_auc, svm_val_auc, svm_best_c = train_linear_svm(
        X_train, y_train, X_val, y_val, seed=seed
    )
    svm_path = models_dir / f"svm_seed{seed}.pkl"
    joblib.dump(svm_model, svm_path)
    print(f"SVM | Seed {seed:3d} | Train AUROC: {svm_tr_auc:.4f} | Val AUROC: {svm_val_auc:.4f} | Best C: {svm_best_c}")

    # 3. PyTorch MLP
    mlp_model, mlp_tr_auc, mlp_val_auc, mlp_hp = train_mlp_pytorch(
        X_train, y_train, X_val, y_val, seed=seed
    )
    mlp_path = models_dir / f"mlp_seed{seed}.pt"
    torch.save(mlp_model.state_dict(), mlp_path)
    print(f"MLP | Seed {seed:3d} | Train AUROC: {mlp_tr_auc:.4f} | Val AUROC: {mlp_val_auc:.4f} | Epoch: {mlp_hp['best_epoch']}")

    return {
        "seed": seed,
        "lr": {"train_auroc": lr_tr_auc, "val_auroc": lr_val_auc, "best_c": lr_best_c},
        "svm": {"train_auroc": svm_tr_auc, "val_auroc": svm_val_auc, "best_c": svm_best_c},
        "mlp": {"train_auroc": mlp_tr_auc, "val_auroc": mlp_val_auc, **mlp_hp},
    }


def train_all_baselines() -> Dict[int, Any]:
    """Train all baselines across seeds [42, 123, 456]."""
    exp_cfg = load_experiment_config()
    seeds = exp_cfg.get("seeds", [42, 123, 456])

    repo_dir = Path(__file__).resolve().parent.parent
    splits_dir = repo_dir / "splits" / "saved"
    scalers_dir = repo_dir / "features" / "feature_cache"
    models_dir = repo_dir / "models" / "saved"

    models_dir.mkdir(parents=True, exist_ok=True)
    scalers_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("TRAINING BASELINE MODELS (LR, SVM, MLP) ON OASIS-1")
    print("=" * 65)
    print(f"Seeds: {seeds}")

    df_oasis, _ = load_oasis1(verbose=False)
    all_results = {}

    for s in seeds:
        print(f"\n--- Seed {s} ---")
        res = run_training_for_seed(s, df_oasis, splits_dir, scalers_dir, models_dir)
        all_results[s] = res

    print("\n" + "=" * 65)
    print("ALL BASELINES TRAINED & SAVED SUCCESSFULLY.")
    print("=" * 65)
    return all_results


if __name__ == "__main__":
    train_all_baselines()
