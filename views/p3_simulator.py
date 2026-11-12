"""
Page 3: What-if Simulator
Allows users to simulate atmospheric conditions with interactive sliders,
computes ML predictions, and provides physical meteorological interpretations.
"""

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    DEFAULT_SYSTEM_KWP,
    DEFAULT_PERFORMANCE_RATIO,
    DEFAULT_EMISSION_FACTOR,
)
from utils.features import build_single_day_features
from utils.models import (
    predict_all_models,
    calculate_energy_kwh,
    calculate_co2_avoided_tonnes,
)


def render():
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; color: #1E293B;">🎛️ What-if Weather & Solar Simulator</h2>
            <p style="color: #64748B; margin-top: 4px; font-size: 0.95rem;">
                Adjust meteorological sliders to see how temperature swings, humidity, and rainfall impact solar potential at Chengalpattu.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "⚠️ **Simulation Notice**: The controls below allow you to test hypothetical meteorological scenarios. "
        "These are synthetic inputs passed directly into the trained ML pipelines, not real-time sensor measurements."
    )

    # Preset Scenarios
    st.markdown("#### Quick Scenario Presets")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    # Initialize session state for sliders if not set
    if "sim_t2m" not in st.session_state:
        st.session_state.sim_t2m = 29.5
        st.session_state.sim_min = 24.0
        st.session_state.sim_max = 34.0
        st.session_state.sim_rh = 68.0
        st.session_state.sim_ws = 2.4
        st.session_state.sim_rain = 0.0
        st.session_state.sim_date = datetime.date(2024, 4, 15)

    if col_p1.button("☀️ Summer Heatwave (May)", use_container_width=True):
        st.session_state.sim_t2m = 35.0
        st.session_state.sim_min = 28.0
        st.session_state.sim_max = 41.0
        st.session_state.sim_rh = 52.0
        st.session_state.sim_ws = 3.2
        st.session_state.sim_rain = 0.0
        st.session_state.sim_date = datetime.date(2024, 5, 20)
        st.rerun()

    if col_p2.button("🌧️ NE Monsoon Storm (Nov)", use_container_width=True):
        st.session_state.sim_t2m = 24.5
        st.session_state.sim_min = 23.0
        st.session_state.sim_max = 26.5
        st.session_state.sim_rh = 94.0
        st.session_state.sim_ws = 6.8
        st.session_state.sim_rain = 55.0
        st.session_state.sim_date = datetime.date(2024, 11, 15)
        st.rerun()

    if col_p3.button("⛅ Mild Winter Day (Jan)", use_container_width=True):
        st.session_state.sim_t2m = 24.0
        st.session_state.sim_min = 19.5
        st.session_state.sim_max = 29.0
        st.session_state.sim_rh = 68.0
        st.session_state.sim_ws = 2.0
        st.session_state.sim_rain = 0.0
        st.session_state.sim_date = datetime.date(2024, 1, 15)
        st.rerun()

    if col_p4.button("🌀 Cyclone Squall (Oct)", use_container_width=True):
        st.session_state.sim_t2m = 26.5
        st.session_state.sim_min = 25.0
        st.session_state.sim_max = 28.0
        st.session_state.sim_rh = 97.0
        st.session_state.sim_ws = 11.5
        st.session_state.sim_rain = 85.0
        st.session_state.sim_date = datetime.date(2024, 10, 28)
        st.rerun()

    st.markdown("---")

    # Interactive Sliders Grid
    col_left, col_right = st.columns(2)

    with col_left:
        sim_date = st.date_input(
            "Calendar Date (Controls Seasonal Solar Orbit / DOY Cycles)",
            value=st.session_state.sim_date,
            help="Determines doy_sin and doy_cos to model solar zenith and day-length changes across the year.",
        )
        sim_t2m = st.slider(
            "Average Temperature (T2M, °C)",
            min_value=15.0,
            max_value=45.0,
            value=float(st.session_state.sim_t2m),
            step=0.5,
        )
        sim_min = st.slider(
            "Minimum Daily Temperature (T2M_MIN, °C)",
            min_value=12.0,
            max_value=35.0,
            value=float(st.session_state.sim_min),
            step=0.5,
        )
        sim_max = st.slider(
            "Maximum Daily Temperature (T2M_MAX, °C)",
            min_value=float(sim_min),
            max_value=50.0,
            value=float(max(sim_min + 1.0, st.session_state.sim_max)),
            step=0.5,
        )

    with col_right:
        sim_rh = st.slider(
            "Relative Humidity (RH2M, %)",
            min_value=10.0,
            max_value=100.0,
            value=float(st.session_state.sim_rh),
            step=1.0,
        )
        sim_ws = st.slider(
            "Wind Speed (WS2M, m/s)",
            min_value=0.2,
            max_value=20.0,
            value=float(st.session_state.sim_ws),
            step=0.2,
        )
        sim_rain = st.slider(
            "Precipitation (PRECTOTCORR, mm/day)",
            min_value=0.0,
            max_value=150.0,
            value=float(st.session_state.sim_rain),
            step=1.0,
        )

        t_range = sim_max - sim_min
        st.metric(
            label="Calculated Diurnal Range (T_RANGE = T_MAX - T_MIN)",
            value=f"{t_range:.1f} °C",
            help="Key engineered feature: wide temperature swings indicate unclouded radiative conditions.",
        )

    # Build features & run inference
    feat_df = build_single_day_features(
        t2m=sim_t2m,
        t2m_max=sim_max,
        t2m_min=sim_min,
        rh2m=sim_rh,
        ws2m=sim_ws,
        prectotcorr=sim_rain,
        date_val=sim_date,
    )
    preds = predict_all_models(feat_df, feat_df.index).iloc[0]

    st.markdown("---")
    st.subheader("💡 Simulated Model Predictions")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(
            label="SVR (Best Model)",
            value=f"{preds['SVR']:.2f} kWh/m²",
            help="Predicted daily solar irradiance on horizontal ground surface.",
        )
    with m_col2:
        st.metric(
            label="Random Forest",
            value=f"{preds['Random Forest']:.2f} kWh/m²",
        )
    with m_col3:
        st.metric(
            label="Linear Regression",
            value=f"{preds['Linear Regression']:.2f} kWh/m²",
        )
    with m_col4:
        st.metric(
            label="Climatology Baseline",
            value=f"{preds['Baseline (climatology)']:.2f} kWh/m²",
            help="Historical day-of-year average, ignoring current weather inputs.",
        )

    # Clean Energy and Carbon offset under this simulation
    best_pred_irr = preds["SVR"]
    daily_energy = calculate_energy_kwh(
        best_pred_irr, DEFAULT_SYSTEM_KWP, DEFAULT_PERFORMANCE_RATIO
    )
    daily_co2_kg = (
        calculate_co2_avoided_tonnes(daily_energy, DEFAULT_EMISSION_FACTOR)
        * 1000.0
    )

    st.info(
        f"⚡ **Estimated 5 kWp Rooftop Yield**: **{daily_energy:.1f} kWh/day** • "
        f"**CO₂ Avoided**: **{daily_co2_kg:.2f} kg/day**"
    )

    # Bar chart of model predictions
    fig = go.Figure(
        data=[
            go.Bar(
                x=[
                    "Baseline (climatology)",
                    "Linear Regression",
                    "SVR",
                    "Random Forest",
                ],
                y=[
                    preds["Baseline (climatology)"],
                    preds["Linear Regression"],
                    preds["SVR"],
                    preds["Random Forest"],
                ],
                marker=dict(color=["#94A3B8", "#FB923C", "#2563EB", "#059669"]),
                text=[
                    f"{preds[m]:.2f}"
                    for m in [
                        "Baseline (climatology)",
                        "Linear Regression",
                        "SVR",
                        "Random Forest",
                    ]
                ],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Simulated Irradiance by Model",
        yaxis_title="Predicted Irradiance (kWh/m²/day)",
        template="plotly_white",
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Plain-English Physical Interpretation
    st.markdown("### 🔬 Plain-English Physical Interpretation")

    interp_points = []

    # Rain & Humidity
    if sim_rain > 10.0 and sim_rh > 85.0:
        interp_points.append(
            f"🌧️ **Monsoon Cloud Attenuation**: Severe rainfall ({sim_rain:.0f} mm/day) and high humidity ({sim_rh:.0f}%) "
            f"strongly indicate thick, overcast stratus/nimbostratus clouds. Direct sunlight is attenuated, resulting in low irradiance "
            f"({best_pred_irr:.2f} kWh/m²/day)."
        )
    elif sim_rain == 0.0 and sim_rh < 65.0:
        interp_points.append(
            f"☀️ **Dry & Clear Atmospheric Transmission**: Zero precipitation and moderate humidity ({sim_rh:.0f}%) minimize "
            f"aerosol scattering and cloud albedo, allowing strong ground-level solar influx ({best_pred_irr:.2f} kWh/m²/day)."
        )

    # Diurnal Range
    if t_range > 10.0:
        interp_points.append(
            f"🌡️ **Wide Diurnal Temperature Swing ({t_range:.1f}°C)**: A large gap between min ({sim_min:.1f}°C) and max ({sim_max:.1f}°C) "
            f"is a strong physical indicator of cloud-free skies. The ground cools rapidly by radiative loss at night and heats up unobstructed during the day."
        )
    elif t_range < 4.0:
        interp_points.append(
            f"☁️ **Narrow Diurnal Temperature Swing ({t_range:.1f}°C)**: Clouds act as an insulating blanket, trapping nighttime heat and reflecting daytime solar insolation, which suppresses the temperature range."
        )

    # Wind
    if sim_ws > 8.0:
        interp_points.append(
            f"💨 **High Wind Speed ({sim_ws:.1f} m/s)**: Typical of tropical depression/cyclone feeder bands in Coastal Tamil Nadu, bringing rapid cloud advection."
        )

    # Seasonal geometry
    interp_points.append(
        f"📅 **Solar Declination for {sim_date.strftime('%B')}**: Day-of-year orbital position encodes solar elevation angles "
        f"(doy_sin={feat_df['doy_sin'].iloc[0]:.2f}, doy_cos={feat_df['doy_cos'].iloc[0]:.2f}), setting the seasonal ceiling for Chengalpattu's latitude (12.69°N)."
    )

    for pt in interp_points:
        st.markdown(f"- {pt}")
