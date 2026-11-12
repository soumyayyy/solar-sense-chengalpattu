"""
Page 6: Solar & CO₂ Calculator
Converts solar irradiance into AC electricity, calculates grid greenhouse gas abatement,
and computes ecological equivalents (trees planted, coal displaced).
Course Unit V: Climate Mitigation.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    DEFAULT_SYSTEM_KWP,
    DEFAULT_PERFORMANCE_RATIO,
    DEFAULT_EMISSION_FACTOR,
    TREE_ANNUAL_CO2_TONNES,
)
from utils.models import (
    load_test_predictions_df,
    calculate_energy_kwh,
    calculate_co2_avoided_tonnes,
    calculate_tree_equivalent,
)


def render():
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; color: #1E293B;">🌱 Solar & CO₂ Climate Impact Calculator</h2>
            <p style="color: #64748B; margin-top: 4px; font-size: 0.95rem;">
                Translate irradiance predictions into clean electricity generation, avoided coal emissions, and ecological benefits.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    test_preds_df = load_test_predictions_df().copy()

    # User Input Parameter Panel
    st.markdown("### 1. Solar Plant & Grid Parameters")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        system_kwp = st.number_input(
            "System Size (kWp)",
            min_value=0.5,
            max_value=1000.0,
            value=float(DEFAULT_SYSTEM_KWP),
            step=0.5,
            help="Rated direct current (DC) peak capacity of the solar PV array in kilowatts.",
        )

    with c2:
        pr = st.number_input(
            "Performance Ratio (PR)",
            min_value=0.40,
            max_value=0.95,
            value=float(DEFAULT_PERFORMANCE_RATIO),
            step=0.01,
            help="System derating factor accounting for inverter losses, cabling resistance, soiling, and temperature coefficient (standard benchmark: 0.75).",
        )

    with c3:
        ef = st.number_input(
            "Grid Emission Factor (tCO₂/MWh)",
            min_value=0.10,
            max_value=1.50,
            value=float(DEFAULT_EMISSION_FACTOR),
            step=0.01,
            help="Central Electricity Authority (CEA) grid baseline emission factor for India (national average ~0.71 tCO₂/MWh).",
        )

    with c4:
        period_choice = st.selectbox(
            "Evaluation Period",
            options=["Combined Test Period (2023–2024)", "Year 2023 Only", "Year 2024 Only"],
            index=0,
        )

    # Filter by period
    if period_choice == "Year 2023 Only":
        df_period = test_preds_df[test_preds_df.index.year == 2023]
    elif period_choice == "Year 2024 Only":
        df_period = test_preds_df[test_preds_df.index.year == 2024]
    else:
        df_period = test_preds_df

    years_count = len(df_period) / 365.25

    # Formulas as defined in notebook Cell 30:
    # Energy = Irradiance * kWp * PR (kWh/day)
    # CO2 = Energy(MWh) * EF = (Energy(kWh) / 1000) * EF
    actual_irr = df_period["ALLSKY_SFC_SW_DWN"]
    pred_irr = df_period["SVR"]  # Best model

    actual_kwh = actual_irr * system_kwp * pr
    pred_kwh = pred_irr * system_kwp * pr

    total_actual_kwh = float(actual_kwh.sum())
    total_pred_kwh = float(pred_kwh.sum())

    annual_actual_kwh = total_actual_kwh / years_count
    annual_pred_kwh = total_pred_kwh / years_count

    actual_co2_t = calculate_co2_avoided_tonnes(annual_actual_kwh, ef)
    pred_co2_t = calculate_co2_avoided_tonnes(annual_pred_kwh, ef)

    trees_actual = calculate_tree_equivalent(actual_co2_t)
    trees_pred = calculate_tree_equivalent(pred_co2_t)

    pct_diff = ((annual_pred_kwh - annual_actual_kwh) / annual_actual_kwh) * 100

    st.markdown("---")
    st.subheader("2. Annual Climate & Generation Outputs (per Year)")

    o1, o2, o3, o4 = st.columns(4)

    with o1:
        st.metric(
            label="Annual Clean Energy Generated",
            value=f"{annual_pred_kwh:,.0f} kWh",
            delta=f"Actual: {annual_actual_kwh:,.0f} kWh ({pct_diff:+.2f}%)",
            help="Annual alternating current (AC) output delivered by the PV system.",
        )

    with o2:
        st.metric(
            label="CO₂ Emissions Avoided",
            value=f"{pred_co2_t:.2f} t CO₂/yr",
            delta=f"Actual: {actual_co2_t:.2f} tonnes",
            help="Total carbon dioxide avoided by replacing coal-fired grid electricity.",
        )

    with o3:
        st.metric(
            label="Equivalent Mature Trees",
            value=f"~{trees_pred:,.0f} trees",
            delta=f"Actual: ~{trees_actual:,.0f} trees",
            help="Based on an average mature tree sequestering 21.77 kg (0.0218 tonnes) of CO₂ per year.",
        )

    with o4:
        coal_displaced_kg = annual_pred_kwh * 0.45
        st.metric(
            label="Coal Combustion Avoided",
            value=f"{coal_displaced_kg/1000:.2f} tonnes",
            delta="Thermal grid offset",
            help="Average Indian subcritical thermal plants burn ~0.45 kg of coal per kWh generated.",
        )

    st.markdown("---")

    # Monthly Bar Chart: Actual vs Predicted Energy
    st.subheader("3. Monthly Energy Profile: Actual vs Model Prediction")
    st.caption("Aggregated monthly clean electricity generation normalized per year.")

    df_monthly = pd.DataFrame(
        {
            "Month": df_period.index.month,
            "Actual_kWh": actual_kwh,
            "Predicted_kWh": pred_kwh,
        }
    )

    monthly_summary = (
        df_monthly.groupby("Month").sum() / years_count
    ).reset_index()
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    monthly_summary["Month_Name"] = monthly_summary["Month"].map(
        lambda m: month_names[m - 1]
    )

    fig_bar = go.Figure()

    fig_bar.add_trace(
        go.Bar(
            x=monthly_summary["Month_Name"],
            y=monthly_summary["Actual_kWh"],
            name="Actual Generation (NASA)",
            marker=dict(color="#F59E0B"),
            text=[f"{v:,.0f}" for v in monthly_summary["Actual_kWh"]],
            textposition="outside",
        )
    )

    fig_bar.add_trace(
        go.Bar(
            x=monthly_summary["Month_Name"],
            y=monthly_summary["Predicted_kWh"],
            name="Predicted Generation (SVR)",
            marker=dict(color="#2563EB"),
            text=[f"{v:,.0f}" for v in monthly_summary["Predicted_kWh"]],
            textposition="outside",
        )
    )

    fig_bar.update_layout(
        title=f"{system_kwp} kWp Rooftop System: Monthly Generation at Chengalpattu",
        xaxis_title="Month",
        yaxis_title="Energy Generated (kWh/month)",
        barmode="group",
        template="plotly_white",
        height=420,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Academic & Ecological Assumptions Card
    st.markdown("---")
    st.subheader("4. Methodological Assumptions & Mathematical Formulation")
    st.markdown(
        f"""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; font-size: 0.9rem; line-height: 1.6;">
            <b>Formulas Applied:</b><br>
            • <b>Daily Energy Yield:</b> <code>Energy (kWh) = Irradiance (kWh/m²/day) × Capacity (kWp) × Performance Ratio</code><br>
            • <b>Avoided Greenhouse Gas Emissions:</b> <code>CO₂ Avoided (tonnes) = [Energy (kWh) / 1000] × Emission Factor (tCO₂/MWh)</code><br>
            • <b>Tree Sequestration Equivalence:</b> Assumes <b>1 mature tropical/subtropical tree sequesters ~21.77 kg (0.02177 tonnes) of CO₂ annually</b> 
            (Source: US EPA Greenhouse Gas Equivalencies & Indian Forestry Carbon Sequestration benchmarks).<br>
            • <b>Performance Ratio:</b> Set at <b>{pr}</b>, accounting for standard subtropical rooftop losses including inverter conversion efficiency (~97%), dust and soiling (~3-5%), wiring drop (~2%), and thermal cell temperature derating (~0.4%/°C above 25°C).
        </div>
        """,
        unsafe_allow_html=True,
    )
