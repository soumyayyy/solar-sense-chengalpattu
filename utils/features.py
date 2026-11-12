"""
Feature engineering module for Solar Sense – Chengalpattu.

Must match the exact feature engineering used during model training:
FEATURES = [T2M, T_RANGE, RH2M, WS2M, PRECTOTCORR, doy_sin, doy_cos]
- T_RANGE = T2M_MAX - T2M_MIN
- doy_sin = sin(2*pi*dayofyear / 365.25)
- doy_cos = cos(2*pi*dayofyear / 365.25)
"""

from typing import Union
import datetime
import numpy as np
import pandas as pd
from config import FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the exact engineered feature set required by the trained models.

    Args:
        df: Input DataFrame containing columns:
            ['T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'WS2M', 'PRECTOTCORR']
            Must have a DatetimeIndex or a 'date' column.

    Returns:
        pd.DataFrame containing strictly [FEATURES] in identical order with DatetimeIndex.
    """
    data = df.copy()

    # Ensure DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
            data = data.set_index("date")
        elif "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"])
            data = data.set_index("Date")
        else:
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as exc:
                raise ValueError(
                    f"DataFrame must have a DatetimeIndex or 'date' column: {exc}"
                )

    required_raw = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PRECTOTCORR"]
    missing = [c for c in required_raw if c not in data.columns]
    if missing:
        raise KeyError(f"Missing required raw weather columns: {missing}")

    # Compute Diurnal Temperature Range
    data["T_RANGE"] = data["T2M_MAX"] - data["T2M_MIN"]

    # Compute seasonal cyclical encoding from day-of-year
    # Uses 365.25 to account for leap years as in notebook training
    doy = data.index.dayofyear
    data["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    data["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    # Return exactly the features in specified training order
    return data[FEATURES].astype(float)


def build_single_day_features(
    t2m: float,
    t2m_max: float,
    t2m_min: float,
    rh2m: float,
    ws2m: float,
    prectotcorr: float,
    date_val: Union[datetime.date, datetime.datetime, str, pd.Timestamp],
) -> pd.DataFrame:
    """
    Build single-row feature DataFrame from scalar values (e.g. for What-if simulator).
    """
    ts = pd.to_datetime(date_val)
    df_row = pd.DataFrame(
        [
            {
                "T2M": float(t2m),
                "T2M_MAX": float(t2m_max),
                "T2M_MIN": float(t2m_min),
                "RH2M": float(rh2m),
                "WS2M": float(ws2m),
                "PRECTOTCORR": float(prectotcorr),
            }
        ],
        index=pd.DatetimeIndex([ts], name="date"),
    )
    return build_features(df_row)
