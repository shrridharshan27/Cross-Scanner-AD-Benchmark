"""Feature Alignment and Leakage-Controlled Scaling Utilities.

Defines feature mapping from Spanish cohort space to OASIS-1 model feature space
and provides scaling functions ensuring StandardScaler is fitted strictly on the
training partition without leaking validation, test, or external cohort data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_ORDER = ["nWBV", "eTIV", "age", "sex"]
SPANISH_MAP = {
    "brain_vol_ratio": "nWBV",
    "icv_estimate": "eTIV",
    "age": "age",
    "sex": "sex",
}


def load_aligned_features(
    oasis_df: pd.DataFrame,
    spanish_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Align Spanish features to OASIS-1 feature space by renaming.

    Does NOT rescale or normalize here — scaler fitted on train only.

    Returns:
        oasis_X:   np.array shape (N_oasis, 4)
        oasis_y:   np.array shape (N_oasis,)
        spanish_X: np.array shape (N_spanish, 4)
        spanish_y: np.array shape (N_spanish,)
        feature_names: ['nWBV', 'eTIV', 'age', 'sex']
    """
    oasis_X = oasis_df[FEATURE_ORDER].to_numpy(dtype=float)
    oasis_y = oasis_df["label"].to_numpy(dtype=int)

    renamed_spanish = spanish_df.rename(columns=SPANISH_MAP)
    spanish_X = renamed_spanish[FEATURE_ORDER].to_numpy(dtype=float)
    spanish_y = renamed_spanish["label"].to_numpy(dtype=int)

    return oasis_X, oasis_y, spanish_X, spanish_y, FEATURE_ORDER


def fit_and_apply_scaler(
    X_train: np.ndarray,
    *other_arrays: Optional[np.ndarray],
) -> Tuple[StandardScaler, List[np.ndarray]]:
    """Fit StandardScaler strictly on training continuous features (first 3 columns).

    Leaves the 4th column (sex, binary {0, 1}) unscaled.

    Args:
        X_train: Training features array shape (N_train, 4)
        *other_arrays: Additional feature arrays (val, test, external) shape (N_i, 4)

    Returns:
        scaler: Fitted StandardScaler instance (fitted on X_train[:, :3])
        scaled_arrays: List containing [X_train_scaled, *others_scaled]
    """
    scaler = StandardScaler()
    # Fit only continuous features: [nWBV, eTIV, age]
    X_train_cont = scaler.fit_transform(X_train[:, :3])
    X_train_scaled = np.column_stack([X_train_cont, X_train[:, 3]])

    scaled_outputs = [X_train_scaled]

    for arr in other_arrays:
        if arr is not None and len(arr) > 0:
            arr_cont = scaler.transform(arr[:, :3])
            arr_scaled = np.column_stack([arr_cont, arr[:, 3]])
            scaled_outputs.append(arr_scaled)
        else:
            scaled_outputs.append(None)

    return scaler, scaled_outputs


def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Apply fitted scaler to continuous features (first 3 cols), leaving 4th (sex) unscaled.

    Args:
        scaler: Fitted StandardScaler instance.
        X: Feature array shape (N, 4) where cols 0-2 are continuous and col 3 is binary sex.

    Returns:
        X_scaled: Scaled feature array shape (N, 4).
    """
    X_cont = scaler.transform(X[:, :3])
    return np.column_stack([X_cont, X[:, 3]])


def save_scaler(scaler: StandardScaler, filepath: Union[str, Path]) -> Path:
    """Save fitted scaler to disk."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, filepath)
    return filepath


def load_scaler(filepath: Union[str, Path]) -> StandardScaler:
    """Load fitted scaler from disk."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler file not found at: {filepath}")
    return joblib.load(filepath)
