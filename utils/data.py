"""
Data acquisition module: NASA POWER Daily API client with caching,
error handling, and offline fallback to clean_data.csv.
"""

from typing import Tuple, Optional
import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    NASA_POWER_API_BASE,
    NASA_RAW_PARAMETERS,
    LATITUDE,
    LONGITUDE,
    CLEAN_DATA_PATH,
)


@st.cache_data(show_spinner=False)
def load_clean_data_csv() -> pd.DataFrame:
    """
    Load cleaned historical dataset from artifacts/clean_data.csv.
    Ensures DatetimeIndex named 'date'.
    """
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Clean data file not found at {CLEAN_DATA_PATH}")

    df = pd.read_csv(CLEAN_DATA_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
    return df.sort_index()


def get_max_live_date() -> datetime.date:
    """
    NASA POWER has an operational processing lag of ~5-7 days.
    Returns today minus 7 days as the maximum recommended date for live queries.
    """
    return datetime.date.today() - datetime.timedelta(days=7)


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_nasa_power_daily(
    start_date: datetime.date,
    end_date: datetime.date,
    timeout_secs: int = 25,
) -> pd.DataFrame:
    """
    Fetch daily solar and meteorological observations from NASA POWER API.
    Replaces -999 missing values with np.nan and interpolates gaps <= 3 days.

    Args:
        start_date: Start date for query
        end_date: End date for query
        timeout_secs: Network timeout in seconds

    Returns:
        pd.DataFrame with DatetimeIndex and parameters as columns.
    """
    params_str = ",".join(NASA_RAW_PARAMETERS)
    s_str = start_date.strftime("%Y%m%d")
    e_str = end_date.strftime("%Y%m%d")

    url = (
        f"{NASA_POWER_API_BASE}"
        f"?parameters={params_str}"
        f"&community=RE"
        f"&longitude={LONGITUDE}"
        f"&latitude={LATITUDE}"
        f"&start={s_str}"
        f"&end={e_str}"
        f"&format=JSON"
    )

    response = requests.get(url, timeout=timeout_secs)
    response.raise_for_status()
    payload = response.json()

    props = payload.get("properties", {})
    parameter_data = props.get("parameter", {})

    if not parameter_data:
        raise ValueError("NASA POWER API returned empty parameter payload.")

    df = pd.DataFrame(parameter_data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"

    # Replace NASA missing value flag (-999 or -999.0) with NaN
    df = df.replace([-999, -999.0, "-999", "-999.0"], np.nan)

    # Cast to float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Interpolate short gaps (<= 3 days)
    if len(df) > 1:
        df = df.interpolate(method="time", limit=3, limit_direction="both")

    return df.sort_index()


def get_daily_data(
    start_date: datetime.date,
    end_date: datetime.date,
    force_offline: bool = False,
) -> Tuple[pd.DataFrame, bool, str]:
    """
    Retrieve data for date range, trying NASA POWER live API first,
    falling back to clean_data.csv if network/API fails or offline mode requested.

    Returns:
        Tuple of (df, is_offline, status_message)
    """
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    clean_df = load_clean_data_csv()

    if force_offline:
        subset = clean_df.loc[
            (clean_df.index.date >= start_date) & (clean_df.index.date <= end_date)
        ]
        if subset.empty:
            # Fall back to nearest available data
            return (
                pd.DataFrame(),
                True,
                f"Offline data is available from {clean_df.index.min().date()} to {clean_df.index.max().date()}. Selected dates ({start_date} to {end_date}) are outside this range.",
            )
        return (
            subset,
            True,
            f"Offline Mode: loaded {len(subset)} records from local stored archive (clean_data.csv).",
        )

    # Try NASA POWER Live API
    try:
        live_df = fetch_nasa_power_daily(start_date, end_date)
        if live_df.empty or live_df["ALLSKY_SFC_SW_DWN"].isna().all():
            raise ValueError("All irradiance values returned by NASA API were missing (-999).")

        return (
            live_df,
            False,
            f"Live data successfully fetched from NASA POWER API for {start_date} to {end_date} (lat: {LATITUDE}, lon: {LONGITUDE}).",
        )
    except Exception as exc:
        # Fall back to clean_data.csv
        subset = clean_df.loc[
            (clean_df.index.date >= start_date) & (clean_df.index.date <= end_date)
        ]
        if not subset.empty:
            return (
                subset,
                True,
                f"NASA POWER API unavailable ({exc.__class__.__name__}: {exc}). Offline mode – using stored data from clean_data.csv.",
            )
        else:
            # Provide sample recent slice from clean_data if requested dates are outside archive
            max_avail = clean_df.index.max().date()
            min_avail = clean_df.index.min().date()
            return (
                pd.DataFrame(),
                True,
                f"NASA POWER API unavailable ({exc}). Stored dataset only covers {min_avail} to {max_avail}.",
            )
