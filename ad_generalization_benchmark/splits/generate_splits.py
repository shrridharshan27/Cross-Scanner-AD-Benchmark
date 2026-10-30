"""Participant-Level Split Generation (Zero Leakage).

Generates stratified participant-level splits for OASIS-1 and Spanish cohorts
across prespecified seeds [42, 123, 456]. Enforces participant-level separation
so that no participant ever appears across train, val, or test sets.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ad_generalization_benchmark.configs import load_experiment_config, resolve_dataset_path
from ad_generalization_benchmark.data.loaders.oasis1_loader import load_oasis1
from ad_generalization_benchmark.data.loaders.spanish_loader import load_spanish_clinical


def generate_stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Generate 3-way participant split (train/val/test) stratified by label."""
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    assert "subject_id" in df.columns, "DataFrame must contain 'subject_id'"
    assert "label" in df.columns, "DataFrame must contain 'label'"

    # Ensure unique participants
    assert df["subject_id"].is_unique, "Subject IDs must be strictly unique (participant-level split)"

    # Step 1: Split into train (70%) and temp (30%)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df["label"],
    )

    # Step 2: Split temp into val and test (50% / 50% of the 30% remainder -> 15% / 15%)
    val_relative_ratio = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_relative_ratio),
        random_state=seed,
        stratify=temp_df["label"],
    )

    train_ids = train_df["subject_id"].tolist()
    val_ids = val_df["subject_id"].tolist()
    test_ids = test_df["subject_id"].tolist()

    # Leakage checks: Strict pairwise disjointness
    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)

    assert len(set_train & set_val) == 0, "Leakage detected: train and val overlap"
    assert len(set_train & set_test) == 0, "Leakage detected: train and test overlap"
    assert len(set_val & set_test) == 0, "Leakage detected: val and test overlap"
    assert len(set_train | set_val | set_test) == len(df), "Participant count mismatch in split"

    return {
        "seed": seed,
        "total_subjects": len(df),
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "counts": {
            "train": {"total": len(train_df), "cn": int((train_df["label"] == 0).sum()), "ad": int((train_df["label"] == 1).sum())},
            "val": {"total": len(val_df), "cn": int((val_df["label"] == 0).sum()), "ad": int((val_df["label"] == 1).sum())},
            "test": {"total": len(test_df), "cn": int((test_df["label"] == 0).sum()), "ad": int((test_df["label"] == 1).sum())},
        },
    }


def generate_spanish_trackB_split(
    df_spanish: pd.DataFrame,
    calibration_ratio: float = 0.20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate Track B Spanish split: 20% unlabeled calibration, 80% evaluation."""
    assert df_spanish["subject_id"].is_unique, "Spanish subject IDs must be unique"

    calib_df, eval_df = train_test_split(
        df_spanish,
        test_size=(1.0 - calibration_ratio),
        random_state=seed,
        stratify=df_spanish["label"],
    )

    calib_ids = calib_df["subject_id"].tolist()
    eval_ids = eval_df["subject_id"].tolist()

    set_calib = set(calib_ids)
    set_eval = set(eval_ids)

    assert len(set_calib & set_eval) == 0, "Leakage: Spanish calibration and evaluation overlap"

    return {
        "seed": seed,
        "total_subjects": len(df_spanish),
        "calibration": calib_ids,
        "evaluation": eval_ids,
        "counts": {
            "calibration": {"total": len(calib_df), "cn": int((calib_df["label"] == 0).sum()), "ad": int((calib_df["label"] == 1).sum())},
            "evaluation": {"total": len(eval_df), "cn": int((eval_df["label"] == 0).sum()), "ad": int((eval_df["label"] == 1).sum())},
        },
    }


def generate_all_splits(
    output_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> None:
    """Generate and save all stratified participant splits across seeds [42, 123, 456]."""
    exp_cfg = load_experiment_config()
    seeds = exp_cfg.get("seeds", [42, 123, 456])

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "saved"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    df_oasis_primary, df_oasis_secondary = load_oasis1(verbose=False)
    df_spanish = load_spanish_clinical(verbose=False)

    # Cross-dataset integrity assertion
    oasis_ids = set(df_oasis_primary["subject_id"])
    spanish_ids = set(df_spanish["subject_id"])
    assert len(oasis_ids & spanish_ids) == 0, "Leakage: Subject ID collision between OASIS-1 and Spanish cohorts"

    if verbose:
        print("=" * 60)
        print("GENERATING STRATIFIED PARTICIPANT-LEVEL SPLITS")
        print("=" * 60)

    for seed in seeds:
        # 1. OASIS-1 Primary Split (128 subjects)
        split_p = generate_stratified_split(df_oasis_primary, seed=seed)
        p_path = output_dir / f"oasis1_primary_seed{seed}.json"
        with open(p_path, "w", encoding="utf-8") as f:
            json.dump(split_p, f, indent=2)

        # 2. OASIS-1 Secondary Split (198 subjects)
        split_s = generate_stratified_split(df_oasis_secondary, seed=seed)
        s_path = output_dir / f"oasis1_secondary_seed{seed}.json"
        with open(s_path, "w", encoding="utf-8") as f:
            json.dump(split_s, f, indent=2)

        # 3. Spanish Track B Split (54 subjects)
        split_b = generate_spanish_trackB_split(df_spanish, seed=seed)
        b_path = output_dir / f"spanish_trackB_seed{seed}.json"
        with open(b_path, "w", encoding="utf-8") as f:
            json.dump(split_b, f, indent=2)

        if verbose:
            print(f"\n[Seed {seed}]")
            print(f"  OASIS-1 Primary (128): Train={len(split_p['train'])}, Val={len(split_p['val'])}, Test={len(split_p['test'])}")
            print(f"    Train counts: {split_p['counts']['train']}")
            print(f"    Val counts:   {split_p['counts']['val']}")
            print(f"    Test counts:  {split_p['counts']['test']}")
            print(f"  Saved -> {p_path.name}")
            print(f"  Spanish Track B (54): Calibration={len(split_b['calibration'])}, Eval={len(split_b['evaluation'])}")
            print(f"  Saved -> {b_path.name}")

    if verbose:
        print("\nAll splits successfully generated with ZERO subject overlap.")
        print("=" * 60)


if __name__ == "__main__":
    generate_all_splits(verbose=True)
