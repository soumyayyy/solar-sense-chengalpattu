"""
Page 2: Live Check (Main Page)
Fetches live NASA POWER data, applies exact feature engineering,
runs ML models + baseline, and presents detailed comparison and climate impact.
"""

from typing import Tuple
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    LATITUDE,
    LONGITUDE,
    FEATURES,
    DEFAULT_SYSTEM_KWP,
    DEFAULT_PERFORMANCE_RATIO,
    DEFAULT_EMISSION_FACTOR,
)
from utils.data import get_daily_data, get_max_live_date, load_clean_data_csv
from utils.features import build_features
from utils.models import (
    predict_all_models,
    calculate_energy_kwh,
    calculate_co2_avoided_tonnes,
)


def render():
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; color: #1E293B;">⚡ Live Solar Check</h2>
            <p style="color: #64748B; margin-top: 4px; font-size: 0.95rem;">
                Compare real-world NASA POWER satellite irradiance against ML predictions for Chengalpattu (12.69°N, 79.98°E).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    clean_df = load_clean_data_csv()
    latest_stored_date = clean_df.index.max().date()
    max_live_date = get_max_live_date()

    # User Input Control Panel
    with st.container():
        col_mode, col_offline = st.columns([3, 1])
        with col_mode:
            mode = st.radio(
                "Inspection Mode",
                options=["Single Day Verification", "Date Range (Up to 31 Days)"],
                horizontal=True,
            )
        with col_offline:
            force_offline = st.checkbox(
                "Simulate Offline Mode",
                value=False,
                help="Test offline fallback behavior by loading directly from local stored archive (clean_data.csv).",
            )

        if mode == "Single Day Verification":
            col_date, col_info = st.columns([1, 2])
            with col_date:
                # Default to latest available stored date (or max live date if earlier)
                default_date = min(latest_stored_date, max_live_date)
                selected_date = st.date_input(
                    "Select Date",
                    value=default_date,
                    min_value=datetime.date(2015, 1, 1),
                    max_value=max_live_date,
                    help="NASA POWER data has an operational publishing lag of ~5-7 days.",
                )
            with col_info:
                st.caption(
                    f"ℹ️ **NASA Operational Lag**: Satellite reanalysis feeds typically take 5–7 days to process. "
                    f"Max real-time date is set to **{max_live_date.strftime('%d %b %Y')}**. "
                    f"Stored archive is available from **2015-01-01** to **{latest_stored_date.strftime('%d %b %Y')}**."
                )
            start_date, end_date = selected_date, selected_date

        else:  # Date Range
            col_start, col_end = st.columns(2)
            with col_start:
                default_start = min(latest_stored_date, max_live_date) - datetime.timedelta(days=14)
                start_date = st.date_input(
                    "Start Date",
                    value=default_start,
                    min_value=datetime.date(2015, 1, 1),
                    max_value=max_live_date,
                )
            with col_end:
                default_end = min(latest_stored_date, max_live_date)
                end_date = st.date_input(
                    "End Date",
                    value=default_end,
                    min_value=datetime.date(2015, 1, 1),
                    max_value=max_live_date,
                )

            # Enforce 31 days max
            if (end_date - start_date).days > 31:
                st.warning("⚠️ Maximum allowable date range is 31 days. Automatically adjusting end date.")
                end_date = start_date + datetime.timedelta(days=31)

            if start_date > end_date:
                st.error("Start date cannot be after end date.")
                return

    # Data Fetching with Spinner
    with st.spinner("Connecting to NASA POWER and running ML models..."):
        data_df, is_offline, status_msg = get_daily_data(
            start_date, end_date, force_offline=force_offline
        )

    # Status Banner
    if is_offline:
        st.warning(f"📡 **Offline Mode Active**: {status_msg}")
    else:
        st.success(f"🌐 **Live API Active**: {status_msg}")

    if data_df.empty:
        st.error(
            "❌ No data returned for the selected dates. Please pick dates within the verified archive "
            f"(2015-01-01 to {latest_stored_date.strftime('%Y-%m-%d')}) or try again later."
        )
        return

    # Build exact features and run all models
    try:
        X_feats = build_features(data_df)
        preds_df = predict_all_models(X_feats, data_df.index)
    except Exception as exc:
        st.error(f"Error computing features or model inference: {exc}")
        return

    # Merge actual target with predictions
    results_df = data_df[["ALLSKY_SFC_SW_DWN"]].rename(
        columns={"ALLSKY_SFC_SW_DWN": "Actual"}
    ).join(preds_df)

    # ---------------- SINGLE DAY VIEW ----------------
    if mode == "Single Day Verification":
        row = results_df.iloc[0]
        actual_val = row["Actual"]

        st.markdown("---")
        st.subheader(f"📊 Results for {start_date.strftime('%A, %d %B %Y')}")

        # Metrics Overview Row
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.metric(
                label="Actual NASA Irradiance",
                value=f"{actual_val:.2f} kWh/m²",
                help="True ground-level solar irradiance measured/reanalyzed by NASA POWER.",
            )

        with m_col2:
            best_model_name = "SVR"
            svr_pred = row["SVR"]
            svr_err = abs(svr_pred - actual_val)
            st.metric(
                label=f"SVR Prediction (Best ML)",
                value=f"{svr_pred:.2f} kWh/m²",
                delta=f"Error: {svr_pred - actual_val:+.2f} ({svr_err:.2f})",
                delta_color="inverse",
            )

        with m_col3:
            rf_pred = row["Random Forest"]
            rf_err = abs(rf_pred - actual_val)
            st.metric(
                label="Random Forest Prediction",
                value=f"{rf_pred:.2f} kWh/m²",
                delta=f"Error: {rf_pred - actual_val:+.2f} ({rf_err:.2f})",
                delta_color="inverse",
            )

        with m_col4:
            base_pred = row["Baseline (climatology)"]
            base_err = abs(base_pred - actual_val)
            st.metric(
                label="Climatology Baseline",
                value=f"{base_pred:.2f} kWh/m²",
                delta=f"Error: {base_pred - actual_val:+.2f} ({base_err:.2f})",
                delta_color="inverse",
                help="Historical day-of-year average (2015-2022). ML models should beat this baseline.",
            )

        # Comparison Table with Best Model Highlight
        table_rows = []
        models_list = [
            "Baseline (climatology)",
            "Linear Regression",
            "SVR",
            "Random Forest",
        ]

        for m_name in models_list:
            pred_v = row[m_name]
            abs_err = abs(pred_v - actual_val)
            pct_err = (abs_err / actual_val * 100) if actual_val > 0 else 0
            energy_kwh = calculate_energy_kwh(
                pred_v, DEFAULT_SYSTEM_KWP, DEFAULT_PERFORMANCE_RATIO
            )
            co2_kg = (
                calculate_co2_avoided_tonnes(energy_kwh, DEFAULT_EMISSION_FACTOR)
                * 1000.0
            )

            table_rows.append(
                {
                    "Model": m_name,
                    "Predicted Irradiance (kWh/m²/day)": round(pred_v, 3),
                    "Absolute Error (kWh/m²)": round(abs_err, 3),
                    "Relative Error (%)": f"{pct_err:.1f}%",
                    "Est. Generation (5 kWp)": f"{energy_kwh:.2f} kWh/day",
                    "CO₂ Avoided": f"{co2_kg:.2f} kg/day",
                }
            )

        summary_table = pd.DataFrame(table_rows).sort_values(
            by="Absolute Error (kWh/m²)"
        )
        best_winner = summary_table.iloc[0]["Model"]

        st.markdown(
            f"""
            <div style="padding: 10px 14px; background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; margin: 16px 0;">
                🏅 <b>Best Performing Model for this day:</b> <span style="color: #065F46; font-weight: bold;">{best_winner}</span> 
                with lowest absolute error of <b>{summary_table.iloc[0]['Absolute Error (kWh/m²)']} kWh/m²</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        # Visual Comparison Bar Chart
        fig = go.Figure()
        chart_models = ["Actual"] + models_list
        chart_vals = [actual_val] + [row[m] for m in models_list]
        colors = ["#F59E0B", "#94A3B8", "#60A5FA", "#3B82F6", "#10B981"]

        fig.add_trace(
            go.Bar(
                x=chart_models,
                y=chart_vals,
                marker=dict(color=colors),
                text=[f"{v:.2f}" for v in chart_vals],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=f"Actual vs Predicted Solar Irradiance ({start_date})",
            yaxis_title="Irradiance (kWh/m²/day)",
            xaxis_title="Source / Model",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- DATE RANGE VIEW ----------------
    else:
        st.markdown("---")
        st.subheader(
            f"📈 Time-Series Verification: {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')} ({len(results_df)} days)"
        )

        # Plotly Multi-line Time Series
        fig = go.Figure()

        # Actual Irradiance
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=results_df["Actual"],
                mode="lines+markers",
                name="Actual (NASA POWER)",
                line=dict(color="#F59E0B", width=3),
                marker=dict(size=6),
            )
        )

        # Climatology Baseline
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=results_df["Baseline (climatology)"],
                mode="lines",
                name="Baseline (Climatology)",
                line=dict(color="#94A3B8", width=2, dash="dash"),
            )
        )

        # Linear Regression
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=results_df["Linear Regression"],
                mode="lines",
                name="Linear Regression",
                line=dict(color="#FB923C", width=1.5),
            )
        )

        # SVR
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=results_df["SVR"],
                mode="lines",
                name="SVR (Support Vector Regressor)",
                line=dict(color="#2563EB", width=2.5),
            )
        )

        # Random Forest
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=results_df["Random Forest"],
                mode="lines",
                name="Random Forest",
                line=dict(color="#059669", width=2),
            )
        )

        fig.update_layout(
            title="Daily Solar Irradiance: Reality vs Machine Learning Predictions",
            xaxis_title="Date",
            yaxis_title="Irradiance (kWh/m²/day)",
            template="plotly_white",
            hovermode="x unified",
            height=450,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Aggregate Errors over Range
        range_errs = []
        for m in [
            "Baseline (climatology)",
            "Linear Regression",
            "SVR",
            "Random Forest",
        ]:
            mae = float(np.mean(np.abs(results_df["Actual"] - results_df[m])))
            rmse = float(
                np.sqrt(np.mean((results_df["Actual"] - results_df[m]) ** 2))
            )
            range_errs.append({"Model": m, "Window MAE": round(mae, 3), "Window RMSE": round(rmse, 3)})

        st.markdown("#### Range Error Summary")
        range_err_df = pd.DataFrame(range_errs).sort_values(by="Window MAE")
        st.dataframe(range_err_df, use_container_width=True, hide_index=True)

    # ---------------- EXPANDABLE WEATHER DATA & DOWNLOAD ----------------
    with st.expander("🔍 Inspect Underlying NASA Weather Observations & Engineered Features"):
        st.markdown(
            "These are the exact atmospheric variables collected by NASA satellites and passed through our feature engineering pipeline:"
        )
        combined_view = data_df.join(preds_df)
        st.dataframe(combined_view, use_container_width=True)

    # Download CSV
    csv_bytes = results_df.to_csv().encode("utf-8")
    st.download_button(
        label="📥 Download Prediction Results (CSV)",
        data=csv_bytes,
        file_name=f"solar_sense_chengalpattu_{start_date}_{end_date}.csv",
        mime="text/csv",
    )
