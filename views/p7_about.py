"""
Page 7: About & Data
Comprehensive project documentation, NASA POWER data provenance, variables dictionary,
data-quality validation summary, academic limitations, formal references, and raw downloads.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from config import (
    COURSE_CODE,
    COURSE_NAME,
    LOCATION_NAME,
    LATITUDE,
    LONGITUDE,
    VARIABLE_DESCRIPTIONS,
    CLEAN_DATA_PATH,
    METRICS_PATH,
    TEST_PREDICTIONS_PATH,
)


def render():
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; color: #1E293B;">ℹ️ About & Data Provenance</h2>
            <p style="color: #64748B; margin-top: 4px; font-size: 0.95rem;">
                Methodology, meteorological data definitions, quality assurance, limitations, and academic citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Academic & Course Affiliation
    st.markdown(
        f"""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 24px;">
            <h4 style="margin: 0 0 8px 0; color: #1E293B;">🎓 Course Context</h4>
            <p style="margin: 0; font-size: 0.92rem; color: #334155; line-height: 1.5;">
                This web application and underlying machine learning research were developed for 
                <b>{COURSE_CODE} – {COURSE_NAME}</b>.<br>
                <b>Syllabus Connections:</b><br>
                • <b>Unit V – Mitigation Strategies:</b> Clean technology deployment, rooftop solar energy estimation, and carbon dioxide abatement.<br>
                • <b>Unit II – Indian Climate & Weather Systems:</b> Tropical monsoon mechanics, insolation variability, cloud optical depth, and diurnal radiative heating at Chengalpattu ({LATITUDE}°N, {LONGITUDE}°E).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Data Source & NASA POWER Info
    st.subheader("1. Data Source: NASA POWER Project")
    st.markdown(
        """
        The solar and meteorological observations in this project are acquired from the 
        [NASA Prediction Of Worldwide Energy Resources (POWER)](https://power.larc.nasa.gov/) API.
        
        POWER integrates global satellite observations (CERES / FLASHFlux) with modern atmospheric assimilation 
        models (Goddard Earth Observing System MERRA-2 reanalysis) on a single unified grid. 
        Parameters are extracted for the single geographic coordinate of **Chengalpattu (12.69°N, 79.98°E)**.
        """
    )

    # Variables Dictionary Table
    st.subheader("2. Meteorological Variables & Feature Dictionary")
    var_rows = []
    for code, info in VARIABLE_DESCRIPTIONS.items():
        role = (
            "Target Variable"
            if code == "ALLSKY_SFC_SW_DWN"
            else (
                "Engineered Feature"
                if code in ["T_RANGE", "doy_sin", "doy_cos"]
                else "Raw Weather Input"
            )
        )
        var_rows.append(
            {
                "Parameter Code": code,
                "Variable Name": info["name"],
                "Unit": info["unit"],
                "Role in Pipeline": role,
                "Description & Physical Meaning": info["desc"],
            }
        )

    var_df = pd.DataFrame(var_rows)
    st.dataframe(var_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Data Quality & Sanity Checks
    st.subheader("3. Data Quality & Preprocessing Protocol")
    st.markdown(
        """
        To guarantee scientific rigor, the raw 10-year dataset (2015–2024, 3,653 calendar days) underwent 
        stringent quality verification:
        
        1. **NASA Sentinel Handling:** NASA uses `-999.0` to denote missing values. All `-999` occurrences were mapped to `NaN`.
        2. **Gap Interpolation:** Isolated missing values spanning $\\le 3$ consecutive days were smoothly filled using time-aware linear interpolation. Zero records were dropped.
        3. **Physical Boundary Checks:**
           - Irradiance constrained to $[0, 10]$ kWh/m²/day (0 violations).
           - Relative humidity constrained to $[0, 100]$% (0 violations).
           - Diurnal sanity verified: $T_{MAX} \\ge T_{MIN}$ across all 3,653 days.
           - Precipitation constrained to $\\ge 0$ mm/day.
        """
    )

    st.markdown("---")

    # Critical Academic Limitations
    st.subheader("4. Critical Limitations & Scope of Applicability")
    st.markdown(
        """
        <div style="background: #FFFBEB; border-left: 4px solid #F59E0B; padding: 14px 18px; border-radius: 6px; font-size: 0.92rem; color: #78350F; line-height: 1.55;">
            <b>Honest limitations must be reported in academic submissions:</b>
            <ol style="margin-top: 6px; padding-left: 20px;">
                <li><b>Spatial Resolution:</b> NASA POWER estimates are derived from a ~0.5° × 0.5° (~50 km × 50 km) satellite/reanalysis grid. Local micro-climates, rooftop rooftop shading, urban heat islands, and tree canopies are not resolved.</li>
                <li><b>Daily Aggregates:</b> The models operate on 24-hour aggregate meteorological variables. Minute-by-minute solar intermittency and rapid cloud ramps cannot be predicted from daily averages.</li>
                <li><b>Fixed Performance Ratio (PR):</b> The energy calculator applies a constant 75% performance ratio. Real-world PV systems experience dynamic derating from dust/soiling accumulation, inverter MPPT clipping, and thermal cell heating above 25°C.</li>
                <li><b>National Grid Emission Factor:</b> The 0.71 tCO₂/MWh emission factor is an annual all-India average published by the CEA. Tamil Nadu possesses high state-level wind and solar penetration, so marginal emission displacements vary by hour and season.</li>
                <li><b>Same-Day Estimation vs. Forecasting:</b> This pipeline performs <i>nowcasting / same-day estimation</i> given measured daily weather variables, rather than forward day-ahead numerical weather prediction (NWP).</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # References
    st.subheader("5. Academic & Institutional References")
    st.markdown(
        """
        1. **NASA POWER Project:** Prediction Of Worldwide Energy Resources, NASA Langley Research Center. Available at: [https://power.larc.nasa.gov](https://power.larc.nasa.gov)
        2. **Central Electricity Authority (CEA):** *CO₂ Baseline Database for the Indian Power Sector (User Guide & Database Version 19)*, Ministry of Power, Government of India, New Delhi, 2024.
        3. **Pedregosa, F., et al.:** *Scikit-learn: Machine Learning in Python*, Journal of Machine Learning Research (JMLR), Vol. 12, pp. 2825–2830, 2011.
        4. **Lundberg, S. M., & Lee, S.-I.:** *A Unified Approach to Interpreting Model Predictions*, Advances in Neural Information Processing Systems (NeurIPS 2017), Vol. 30, 2017.
        5. **Ministry of New and Renewable Energy (MNRE):** *Grid Connected Rooftop Solar Power Programme*, Government of India, 2023.
        """
    )

    st.markdown("---")

    # Artifact Downloads
    st.subheader("6. Download Verified Research Artifacts")
    d_col1, d_col2, d_col3 = st.columns(3)

    if CLEAN_DATA_PATH.exists():
        with open(CLEAN_DATA_PATH, "rb") as f:
            d_col1.download_button(
                label="📥 Download Clean Dataset (CSV)",
                data=f.read(),
                file_name="clean_data_chengalpattu_2015_2024.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "rb") as f:
            d_col2.download_button(
                label="📥 Download Benchmark Metrics (CSV)",
                data=f.read(),
                file_name="model_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if TEST_PREDICTIONS_PATH.exists():
        with open(TEST_PREDICTIONS_PATH, "rb") as f:
            d_col3.download_button(
                label="📥 Download Test Predictions (CSV)",
                data=f.read(),
                file_name="test_predictions_2023_2024.csv",
                mime="text/csv",
                use_container_width=True,
            )
