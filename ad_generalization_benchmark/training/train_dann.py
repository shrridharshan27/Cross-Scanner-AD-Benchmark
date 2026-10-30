"""Track B: Domain-Adversarial Neural Network (DANN) Harmonization Pipeline.

WHAT DANN DOES:
DANN learns feature representations that are simultaneously:
  1. Discriminative of Alzheimer's disease diagnosis (AD vs CN) on the labeled source domain.
  2. Invariant to scanner / domain shifts between OASIS-1 (Siemens 1.5T) and the Spanish cohort.
This is achieved via an adversarial game between a feature extractor and a domain discriminator
connected through a Gradient Reversal Layer (GRL). The feature extractor attempts to MAXIMIZE
domain classification error, while the domain discriminator attempts to MINIMIZE it.

Strict Leakage & Protocol Controls:
- Source (labeled): OASIS-1 train partition (features + AD/CN labels, domain=0).
- Target (unlabeled): Spanish calibration partition (~11 subjects, features only, domain=1).
- Target diagnostic labels are NEVER used in any loss computation, validation, or training step.
- Model selection / early stopping monitors OASIS-1 validation AUROC only.
- Evaluation is performed on OASIS-1 held-out test and the Spanish evaluation partition (~43 subjects).

IMPORTANT SCIENTIFIC WARNING:
Spanish calibration n ~ 11 is small for adversarial neural networks. The domain discriminator
may experience high variance or difficulty learning stable decision boundaries. This is reported
transparently as a known benchmark limitation.
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
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ad_generalization_benchmark.configs import load_experiment_config
from ad_generalization_benchmark.data.loaders.oasis1_loader import load_oasis1
from ad_generalization_benchmark.evaluation.evaluate import (
    compute_bootstrap_auroc_ci,
    find_optimal_threshold,
)
from ad_generalization_benchmark.features.align_features import fit_and_apply_scaler, apply_scaler

FEATURE_ORDER = ["nWBV", "eTIV", "age", "sex"]
SPANISH_MAP = {
    "brain_vol_ratio": "nWBV",
    "icv_estimate": "eTIV",
    "age": "age",
    "sex": "sex",
}


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer autograd implementation."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


class DANNArchitecture(nn.Module):
    """Domain-Adversarial Neural Network matching benchmark specification."""

    def __init__(self, input_dim: int = 4, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

        # Feature Extractor F: Linear(4,32) -> ReLU -> Linear(32,16) -> ReLU
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Label Classifier C: Linear(16,8) -> ReLU -> Linear(8,1) -> Sigmoid
        self.label_classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # Domain Discriminator D: GRL -> Linear(16,8) -> ReLU -> Linear(8,1) -> Sigmoid
        self.domain_discriminator = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, alpha: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if alpha is None:
            alpha = self.alpha
        features = self.feature_extractor(x)
        label_preds = self.label_classifier(features)
        reversed_features = GradientReversalFunction.apply(features, alpha)
        domain_preds = self.domain_discriminator(reversed_features)
        return label_preds, domain_preds

    def predict_label(self, x: torch.Tensor) -> torch.Tensor:
        """Inference path: predict diagnostic probabilities without domain reversal."""
        features = self.feature_extractor(x)
        return self.label_classifier(features)


def train_dann_model(
    source_X: np.ndarray,
    source_y: np.ndarray,
    target_calib_X: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    seed: int,
    lr: float = 1e-3,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 16,
) -> Tuple[DANNArchitecture, float, float, Dict[str, Any]]:
    """Train DANN adversarially on labeled source and unlabeled target calibration data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_neg = int((source_y == 0).sum())
    n_pos = int((source_y == 1).sum())
    pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    model = DANNArchitecture(input_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    source_dataset = TensorDataset(
        torch.tensor(source_X, dtype=torch.float32),
        torch.tensor(source_y, dtype=torch.float32),
    )
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)

    target_tensor = torch.tensor(target_calib_X, dtype=torch.float32)
    val_tensor_x = torch.tensor(val_X, dtype=torch.float32)

    best_val_auroc = -1.0
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0

    n_target = len(target_tensor)

    for epoch in range(1, max_epochs + 1):
        model.train()

        # Dynamic alpha schedule: alpha = 2 / (1 + exp(-10 * epoch / max_epochs)) - 1
        p = float(epoch) / float(max_epochs)
        alpha = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        for src_x, src_y in source_loader:
            optimizer.zero_grad()
            b_src = len(src_x)

            # Sample matching number of target instances with replacement if needed
            t_idx = torch.randint(0, n_target, size=(b_src,))
            tgt_x = target_tensor[t_idx]

            # 1. Forward passes
            src_preds, src_dom_preds = model(src_x, alpha=alpha)
            _, tgt_dom_preds = model(tgt_x, alpha=alpha)

            # 2. Classification Loss (source only, weighted BCE)
            weights = torch.where(src_y == 1.0, torch.tensor(pos_weight), torch.tensor(1.0))
            cls_loss = -(
                weights
                * (
                    src_y * torch.log(src_preds.flatten() + 1e-7)
                    + (1.0 - src_y) * torch.log(1.0 - src_preds.flatten() + 1e-7)
                )
            ).mean()

            # 3. Domain Loss (source domain=0, target domain=1)
            all_dom_preds = torch.cat([src_dom_preds.flatten(), tgt_dom_preds.flatten()])
            all_dom_labels = torch.cat([torch.zeros(b_src), torch.ones(b_src)])
            dom_loss = -(
                all_dom_labels * torch.log(all_dom_preds + 1e-7)
                + (1.0 - all_dom_labels) * torch.log(1.0 - all_dom_preds + 1e-7)
            ).mean()

            # Total adversarial loss
            total_loss = cls_loss + dom_loss
            total_loss.backward()
            optimizer.step()

        # Validation evaluation (source val set)
        model.eval()
        with torch.no_grad():
            val_probs = model.predict_label(val_tensor_x).cpu().numpy().flatten()
            val_auroc = float(roc_auc_score(val_y, val_probs))

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
        src_train_tensor = torch.tensor(source_X, dtype=torch.float32)
        train_probs = model.predict_label(src_train_tensor).cpu().numpy().flatten()
        train_auroc = float(roc_auc_score(source_y, train_probs))

    hp_info = {
        "lr": lr,
        "best_epoch": best_epoch,
        "pos_weight": pos_weight,
        "final_alpha": alpha,
    }
    return model, train_auroc, best_val_auroc, hp_info


def run_dann_for_seed(
    seed: int,
    df_oasis: pd.DataFrame,
    df_spanish: pd.DataFrame,
    splits_dir: Path,
    models_dir: Path,
    results_dir: Path,
) -> Dict[str, Any]:
    """Execute DANN training and evaluation pipeline for a single seed."""
    oasis_split_file = splits_dir / f"oasis1_primary_seed{seed}.json"
    sp_split_file = splits_dir / f"spanish_trackB_seed{seed}.json"

    with open(oasis_split_file, "r", encoding="utf-8") as f:
        oasis_split = json.load(f)
    with open(sp_split_file, "r", encoding="utf-8") as f:
        sp_split = json.load(f)

    train_df = df_oasis[df_oasis["subject_id"].isin(set(oasis_split["train"]))].copy()
    val_df = df_oasis[df_oasis["subject_id"].isin(set(oasis_split["val"]))].copy()
    test_df = df_oasis[df_oasis["subject_id"].isin(set(oasis_split["test"]))].copy()

    calib_df = df_spanish[df_spanish["subject_id"].isin(set(sp_split["calibration"]))].copy()
    eval_df = df_spanish[df_spanish["subject_id"].isin(set(sp_split["evaluation"]))].copy()

    # Raw features
    X_train_raw = train_df[FEATURE_ORDER].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)

    X_val_raw = val_df[FEATURE_ORDER].to_numpy(dtype=float)
    y_val = val_df["label"].to_numpy(dtype=int)

    X_test_raw = test_df[FEATURE_ORDER].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy(dtype=int)

    # Spanish features (mapped and unit aligned)
    X_calib_raw = calib_df.rename(columns=SPANISH_MAP)[FEATURE_ORDER].to_numpy(dtype=float)
    X_eval_raw = eval_df.rename(columns=SPANISH_MAP)[FEATURE_ORDER].to_numpy(dtype=float)
    y_eval = eval_df["label"].to_numpy(dtype=int)

    # Fit scaler strictly on OASIS-1 train partition continuous features
    scaler, [X_train, X_val, X_test, X_calib, X_eval] = fit_and_apply_scaler(
        X_train_raw, X_val_raw, X_test_raw, X_calib_raw, X_eval_raw
    )

    # Train DANN
    model, tr_auc, val_auc, hp_info = train_dann_model(
        X_train, y_train, X_calib, X_val, y_val, seed=seed
    )

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / f"dann_seed{seed}.pt")

    # Predict probabilities
    with torch.no_grad():
        val_probs = model.predict_label(torch.tensor(X_val, dtype=torch.float32)).cpu().numpy().flatten()
        test_probs = model.predict_label(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy().flatten()
        eval_probs = model.predict_label(torch.tensor(X_eval, dtype=torch.float32)).cpu().numpy().flatten()

    # Threshold selected on OASIS validation set
    threshold = find_optimal_threshold(y_val, val_probs)

    # Internal OASIS test metrics
    test_preds = (test_probs >= threshold).astype(int)
    int_auroc = float(roc_auc_score(y_test, test_probs))
    int_ci_low, int_ci_high = compute_bootstrap_auroc_ci(y_test, test_probs, n_bootstraps=1000, seed=seed)
    int_bal_acc = float(balanced_accuracy_score(y_test, test_preds))
    int_sens = float(recall_score(y_test, test_preds, pos_label=1, zero_division=0))
    int_spec = float(recall_score(y_test, test_preds, pos_label=0, zero_division=0))

    # External Spanish eval metrics
    eval_preds = (eval_probs >= threshold).astype(int)
    ext_auroc = float(roc_auc_score(y_eval, eval_probs))
    ext_ci_low, ext_ci_high = compute_bootstrap_auroc_ci(y_eval, eval_probs, n_bootstraps=1000, seed=seed)
    ext_bal_acc = float(balanced_accuracy_score(y_eval, eval_preds))
    ext_sens = float(recall_score(y_eval, eval_preds, pos_label=1, zero_division=0))
    ext_spec = float(recall_score(y_eval, eval_preds, pos_label=0, zero_division=0))

    gap = int_auroc - ext_auroc

    # Load baseline MLP external AUROC for comparison
    base_file = results_dir / f"external_trackA_seed{seed}.json"
    unadapted_sp_auroc = None
    if base_file.exists():
        with open(base_file, "r", encoding="utf-8") as bf:
            unadapted_data = json.load(bf)
            unadapted_sp_auroc = unadapted_data["mlp"]["spanish_auroc"]

    improvement = (ext_auroc - unadapted_sp_auroc) if unadapted_sp_auroc is not None else None

    result = {
        "seed": seed,
        "method": "DANN",
        "threshold": threshold,
        "internal": {
            "auroc": int_auroc,
            "ci_low": int_ci_low,
            "ci_high": int_ci_high,
            "balanced_accuracy": int_bal_acc,
            "sensitivity": int_sens,
            "specificity": int_spec,
            "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
        },
        "external": {
            "spanish_eval_n": len(y_eval),
            "auroc": ext_auroc,
            "ci_low": ext_ci_low,
            "ci_high": ext_ci_high,
            "balanced_accuracy": ext_bal_acc,
            "sensitivity": ext_sens,
            "specificity": ext_spec,
            "confusion_matrix": confusion_matrix(y_eval, eval_preds).tolist(),
        },
        "generalization_gap": gap,
        "unadapted_baseline_sp_auroc": unadapted_sp_auroc,
        "adaptation_improvement": improvement,
        "training_info": hp_info,
    }

    out_file = results_dir / f"dann_eval_seed{seed}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    imp_str = f"{improvement:+.4f}" if improvement is not None else "N/A"
    print(
        f"DANN   | Seed {seed:3d} | Internal AUROC: {int_auroc:.4f} | "
        f"Spanish AUROC: {ext_auroc:.4f} | Improvement: {imp_str}"
    )

    return result


def train_and_eval_dann_all() -> Dict[int, Any]:
    """Train and evaluate DANN across all seeds [42, 123, 456]."""
    exp_cfg = load_experiment_config()
    seeds = exp_cfg.get("seeds", [42, 123, 456])

    repo_dir = Path(__file__).resolve().parent.parent
    splits_dir = repo_dir / "splits" / "saved"
    models_dir = repo_dir / "models" / "saved"
    results_dir = repo_dir / "results"
    sp_cache = repo_dir / "features" / "feature_cache" / "spanish_features.csv"

    print("=" * 75)
    print("TRACK B: DOMAIN-ADVERSARIAL NEURAL NETWORK (DANN) BENCHMARK")
    print("=" * 75)
    print(f"Seeds: {seeds}")

    df_oasis, _ = load_oasis1(verbose=False)
    df_spanish = pd.read_csv(sp_cache)

    all_results = {}
    for s in seeds:
        res = run_dann_for_seed(s, df_oasis, df_spanish, splits_dir, models_dir, results_dir)
        all_results[s] = res

    int_aucs = [r["internal"]["auroc"] for r in all_results.values()]
    ext_aucs = [r["external"]["auroc"] for r in all_results.values()]
    gaps = [r["generalization_gap"] for r in all_results.values()]
    imps = [r["adaptation_improvement"] for r in all_results.values() if r["adaptation_improvement"] is not None]

    print("\n" + "=" * 75)
    print("DANN SUMMARY (Mean ± Std over 3 seeds)")
    print("=" * 75)
    print(f"Internal OASIS AUROC:  {np.mean(int_aucs):.4f} ± {np.std(int_aucs):.4f}")
    print(f"External Spanish AUROC: {np.mean(ext_aucs):.4f} ± {np.std(ext_aucs):.4f}")
    print(f"Generalization Gap:     {np.mean(gaps):+.4f} ± {np.std(gaps):.4f}")
    if imps:
        print(f"Adaptation Improvement: {np.mean(imps):+.4f} ± {np.std(imps):.4f} AUROC points")
    print("=" * 75)

    return all_results


if __name__ == "__main__":
    train_and_eval_dann_all()
