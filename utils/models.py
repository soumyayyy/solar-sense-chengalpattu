"""
Model management module: cached model loading, baseline climatology calculation,
multi-model batch inference, and solar/climate impact conversions.
"""

from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    MODEL_PATHS,
    METRICS_PATH,
    TEST_PREDICTIONS_PATH,
    RF_PERM_IMPORTANCE_PATH,
    TARGET,
    TRAIN_END_YEAR,
    DEFAULT_SYSTEM_KWP,
    DEFAULT_PERFORMANCE_RATIO,
    DEFAULT_EMISSION_FACTOR,
    TREE_ANNUAL_CO2_TONNES,
)
from utils.data import load_clean_data_csv


@st.cache_resource(show_spinner=False)
def load_trained_models() -> Dict[str, Any]:
    """
    Load saved scikit-learn models from artifacts directory.
    Uses st.cache_resource to ensure single memory footprint.
    """
    models = {}
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Model file for {name} not found at {path}")
        models[name] = joblib.load(path)
    return models


@st.cache_data(show_spinner=False)
def get_climatology_baseline() -> Tuple[pd.Series, float]:
    """
    Compute daily climatology baseline from training data (<= 2022):
    Mean ALLSKY_SFC_SW_DWN for each day-of-year (1-366).
    """
    clean_df = load_clean_data_csv()
    train_df = clean_df[clean_df.index.year <= TRAIN_END_YEAR]
    clim = train_df.groupby(train_df.index.dayofyear)[TARGET].mean()
    overall_mean = float(train_df[TARGET].mean())
    return clim, overall_mean


def predict_all_models(
    X_features: pd.DataFrame,
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Generate predictions for all models plus climatological baseline.

    Args:
        X_features: Feature DataFrame with columns matching FEATURES order.
        date_index: DatetimeIndex corresponding to the rows of X_features.

    Returns:
        pd.DataFrame with predictions from:
        ['Baseline (climatology)', 'Linear Regression', 'SVR', 'Random Forest']
    """
    models = load_trained_models()
    clim_series, overall_mean = get_climatology_baseline()

    # Baseline prediction: day-of-year lookup
    doy = pd.Series(date_index.dayofyear, index=date_index)
    baseline_pred = doy.map(clim_series).fillna(overall_mean).values

    # Predictions from trained ML pipelines
    lr_pred = models["Linear Regression"].predict(X_features)
    svr_pred = models["SVR"].predict(X_features)
    rf_pred = models["Random Forest"].predict(X_features)

    # Solar irradiance physically cannot be negative; clip at 0
    df_preds = pd.DataFrame(
        {
            "Baseline (climatology)": np.clip(baseline_pred, 0, None),
            "Linear Regression": np.clip(lr_pred, 0, None),
            "SVR": np.clip(svr_pred, 0, None),
            "Random Forest": np.clip(rf_pred, 0, None),
        },
        index=date_index,
    )
    return df_preds


@st.cache_data(show_spinner=False)
def load_metrics_df() -> pd.DataFrame:
    """Load model evaluation metrics from artifacts/metrics.csv."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found at {METRICS_PATH}")
    return pd.read_csv(METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_test_predictions_df() -> pd.DataFrame:
    """Load test predictions from artifacts/test_predictions.csv."""
    if not TEST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Test predictions file not found at {TEST_PREDICTIONS_PATH}"
        )
    df = pd.read_csv(TEST_PREDICTIONS_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
    return df


@st.cache_data(show_spinner=False)
def load_rf_permutation_importance_df() -> pd.DataFrame:
    """Load Random Forest permutation importance from artifacts/rf_permutation_importance.csv."""
    if not RF_PERM_IMPORTANCE_PATH.exists():
        raise FileNotFoundError(
            f"Importance file not found at {RF_PERM_IMPORTANCE_PATH}"
        )
    df = pd.read_csv(RF_PERM_IMPORTANCE_PATH, header=None)
    # File format is: Feature, Importance_score
    df.columns = ["Feature", "Importance"]
    # Drop empty or header row if present
    df = df[df["Feature"].notna() & (df["Feature"] != "")]
    df["Importance"] = pd.to_numeric(df["Importance"], errors="coerce")
    return df.sort_values(by="Importance", ascending=True)


def calculate_energy_kwh(
    irradiance_kwh_m2: float,
    system_kwp: float = DEFAULT_SYSTEM_KWP,
    performance_ratio: float = DEFAULT_PERFORMANCE_RATIO,
) -> float:
    """
    Energy (kWh) = Irradiance (kWh/m²/day) * system_kwp * performance_ratio
    Formula aligned with course 21CEO310T Unit V (Mitigation).
    """
    return float(irradiance_kwh_m2 * system_kwp * performance_ratio)


def calculate_co2_avoided_tonnes(
    energy_kwh: float,
    emission_factor: float = DEFAULT_EMISSION_FACTOR,
) -> float:
    """
    CO2 avoided (tonnes) = Energy (MWh) * emission_factor
                        = (Energy (kWh) / 1000) * emission_factor
    """
    return float((energy_kwh / 1000.0) * emission_factor)


def calculate_tree_equivalent(co2_tonnes: float) -> float:
    """
    Equivalent mature trees absorbing CO2 per year.
    Assumption: 1 mature tree absorbs ~21.77 kg (0.02177 tonnes) of CO2/year.
    """
    return float(co2_tonnes / TREE_ANNUAL_CO2_TONNES)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculate MAE, MSE, RMSE, R²."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }
