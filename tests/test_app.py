"""
Pytest unit tests for Solar Sense – Chengalpattu.
Covers:
- Feature engineering output order, shape, and calculation accuracy
- NASA -999 missing value conversion and interpolation
- Solar energy, CO2 avoided, and tree equivalence formulas
- Model loading, pipeline presence, and sample inference
"""

import datetime
import numpy as np
import pandas as pd
import pytest

from config import (
    FEATURES,
    DEFAULT_SYSTEM_KWP,
    DEFAULT_PERFORMANCE_RATIO,
    DEFAULT_EMISSION_FACTOR,
    TREE_ANNUAL_CO2_TONNES,
    CLEAN_DATA_PATH,
    METRICS_PATH,
    TEST_PREDICTIONS_PATH,
    RF_PERM_IMPORTANCE_PATH,
    SHAP_SUMMARY_PATH,
)
from utils.features import build_features, build_single_day_features
from utils.models import (
    load_trained_models,
    predict_all_models,
    calculate_energy_kwh,
    calculate_co2_avoided_tonnes,
    calculate_tree_equivalent,
)


def test_feature_engineering_order_and_shape():
    """Verify that build_features outputs exact features in exact required order."""
    dates = pd.date_range("2024-03-20", periods=5, freq="D")
    df_raw = pd.DataFrame(
        {
            "T2M": [28.0, 29.5, 30.0, 27.5, 26.0],
            "T2M_MAX": [34.0, 35.0, 36.0, 32.0, 30.0],
            "T2M_MIN": [22.0, 23.0, 24.0, 21.0, 20.0],
            "RH2M": [65.0, 70.0, 60.0, 80.0, 85.0],
            "WS2M": [2.5, 3.0, 1.8, 4.2, 5.0],
            "PRECTOTCORR": [0.0, 0.0, 0.0, 12.5, 35.0],
        },
        index=dates,
    )

    feat_df = build_features(df_raw)

    # Check exact column ordering
    assert list(feat_df.columns) == FEATURES
    assert len(feat_df) == 5

    # Check T_RANGE calculation
    expected_trange = df_raw["T2M_MAX"] - df_raw["T2M_MIN"]
    np.testing.assert_allclose(feat_df["T_RANGE"].values, expected_trange.values)

    # Check doy_sin and doy_cos values
    doy = dates.dayofyear
    expected_sin = np.sin(2 * np.pi * doy / 365.25)
    expected_cos = np.cos(2 * np.pi * doy / 365.25)
    np.testing.assert_allclose(feat_df["doy_sin"].values, expected_sin)
    np.testing.assert_allclose(feat_df["doy_cos"].values, expected_cos)


def test_single_day_feature_helper():
    """Verify single day helper creates compatible 1-row DataFrame."""
    f = build_single_day_features(
        t2m=30.0,
        t2m_max=36.0,
        t2m_min=24.0,
        rh2m=60.0,
        ws2m=2.5,
        prectotcorr=0.0,
        date_val=datetime.date(2024, 6, 21),
    )
    assert list(f.columns) == FEATURES
    assert len(f) == 1
    assert f["T_RANGE"].iloc[0] == 12.0


def test_missing_value_handling_999():
    """Verify conversion of -999 flags to NaN and graceful interpolation."""
    df_with_999 = pd.DataFrame(
        {
            "T2M": [28.0, -999.0, 30.0],
            "T2M_MAX": [34.0, 35.0, -999],
            "T2M_MIN": [22.0, 23.0, 24.0],
            "RH2M": [65.0, 70.0, 60.0],
            "WS2M": [2.5, -999, 1.8],
            "PRECTOTCORR": [0.0, 0.0, 0.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    # Clean replacing -999 with NaN and interpolating
    cleaned = df_with_999.replace([-999, -999.0], np.nan).interpolate(
        limit=3, limit_direction="both"
    )

    assert not cleaned.isna().any().any()
    assert cleaned.loc["2024-01-02", "T2M"] == 29.0
    assert cleaned.loc["2024-01-03", "T2M_MAX"] == 35.0


def test_energy_and_co2_formulas():
    """Verify solar generation, carbon abatement, and tree equivalents."""
    # 1. Solar energy: 5 kWh/m2 * 5 kWp * 0.75 = 18.75 kWh
    energy = calculate_energy_kwh(
        irradiance_kwh_m2=5.0, system_kwp=5.0, performance_ratio=0.75
    )
    assert energy == 18.75

    # 2. CO2 Avoided: 1,000 kWh = 1 MWh -> 1 * 0.71 = 0.71 tonnes
    co2 = calculate_co2_avoided_tonnes(energy_kwh=1000.0, emission_factor=0.71)
    assert pytest.approx(co2, 0.001) == 0.71

    # 3. Tree equivalence: 0.71 tonnes / 0.02177 t/tree/yr ~ 32.61 trees
    trees = calculate_tree_equivalent(co2)
    assert pytest.approx(trees, 0.01) == (0.71 / TREE_ANNUAL_CO2_TONNES)


def test_model_loading_and_inference():
    """Verify all 3 trained models load cleanly and produce valid non-negative predictions."""
    models = load_trained_models()
    assert "Linear Regression" in models
    assert "SVR" in models
    assert "Random Forest" in models

    # Check each model has predict method
    for name, m in models.items():
        assert hasattr(m, "predict"), f"{name} must have a predict method"

    # Run multi-model prediction on sample input
    sample_feat = build_single_day_features(
        t2m=28.5,
        t2m_max=33.5,
        t2m_min=23.5,
        rh2m=72.0,
        ws2m=2.8,
        prectotcorr=0.0,
        date_val="2024-05-15",
    )
    preds = predict_all_models(sample_feat, sample_feat.index)

    assert "Baseline (climatology)" in preds.columns
    assert "Linear Regression" in preds.columns
    assert "SVR" in preds.columns
    assert "Random Forest" in preds.columns

    # Irradiance cannot be negative and realistically below 10 for Chengalpattu
    for col in preds.columns:
        val = preds[col].iloc[0]
        assert 0.0 <= val <= 10.0, f"{col} prediction {val} out of plausible bounds [0, 10]"


def test_artifacts_exist():
    """Verify all required artifacts are present and non-empty."""
    assert CLEAN_DATA_PATH.exists() and CLEAN_DATA_PATH.stat().st_size > 0
    assert METRICS_PATH.exists() and METRICS_PATH.stat().st_size > 0
    assert TEST_PREDICTIONS_PATH.exists() and TEST_PREDICTIONS_PATH.stat().st_size > 0
    assert (
        RF_PERM_IMPORTANCE_PATH.exists()
        and RF_PERM_IMPORTANCE_PATH.stat().st_size > 0
    )
    assert SHAP_SUMMARY_PATH.exists() and SHAP_SUMMARY_PATH.stat().st_size > 0
