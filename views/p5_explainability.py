"""
Page 5: Explainability
Explains why the models make their predictions using Random Forest permutation importance,
plain-language physical meteorology explanations, and static pre-computed SHAP summary plots.
"""

from pathlib import Path
import plotly.graph_objects as go
import streamlit as st

from config import SHAP_SUMMARY_PATH, VARIABLE_DESCRIPTIONS
from utils.models import load_rf_permutation_importance_df


def render():
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; color: #1E293B;">🔍 Model Explainability & Physical Drivers</h2>
            <p style="color: #64748B; margin-top: 4px; font-size: 0.95rem;">
                Deconstruct the black-box: how atmospheric physics and weather variables drive solar predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        In climate-tech applications, model explainability is as vital as predictive accuracy. 
        Engineers and grid operators must understand **why** an ML model predicts low or high irradiance 
        to trust its recommendations for dispatch and battery storage scheduling.
        """
    )

    # Permutation Importance Section
    st.subheader("1. Feature Importance: Random Forest Permutation Importance")
    st.caption(
        "Permutation importance measures the drop in test-set R² score when a feature's values are randomly shuffled. "
        "A large drop indicates that the model relies heavily on that feature."
    )

    rf_perm_df = load_rf_permutation_importance_df()

    # Add friendly names
    rf_perm_df["Friendly_Name"] = rf_perm_df["Feature"].map(
        lambda f: VARIABLE_DESCRIPTIONS.get(f, {}).get("name", f)
    )

    fig_perm = go.Figure(
        go.Bar(
            x=rf_perm_df["Importance"],
            y=rf_perm_df["Feature"],
            orientation="h",
            marker=dict(color="#2563EB"),
            text=[f"{v:.3f} drop in R²" for v in rf_perm_df["Importance"]],
            textposition="outside",
            customdata=rf_perm_df["Friendly_Name"],
            hovertemplate="<b>%{y}</b> (%{customdata})<br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig_perm.update_layout(
        title="Drop in R² Score on Test Set When Feature is Shuffled",
        xaxis_title="Permutation Importance (Mean R² Drop)",
        yaxis_title="Feature",
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig_perm, use_container_width=True)

    st.markdown("---")

    # Plain-Language Physical Meaning of Every Feature
    st.subheader("2. Meteorological & Physical Translation")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        with st.expander("🌧️ PRECTOTCORR – Rainfall (Rank #1, Drop ~0.290)", expanded=True):
            st.markdown(
                """
                **Why it matters:** Precipitation is the single strongest physical indicator of dense cloud cover. 
                Rain in Chengalpattu occurs almost exclusively underneath cumulonimbus or nimbostratus clouds 
                possessing massive optical depths. When rain occurs, direct-beam sunlight is near zero, and ground irradiance 
                drops to low diffuse levels.
                """
            )

        with st.expander("🌡️ T2M – Average 2m Temperature (Rank #2, Drop ~0.223)", expanded=True):
            st.markdown(
                """
                **Why it matters:** Surface air temperature is directly linked to daytime solar insolation. In tropical 
                Tamil Nadu, clear days generate intense ground radiative absorption that heats the lower troposphere, 
                whereas overcast monsoon days keep temperatures subdued.
                """
            )

        with st.expander("💧 RH2M – Relative Humidity (Rank #3, Drop ~0.113)", expanded=True):
            st.markdown(
                """
                **Why it matters:** Water vapor molecules cause Rayleigh and aerosol Mie scattering of shortwave solar 
                radiation. High humidity indicates a moisture-saturated boundary layer prone to rapid convective cloud 
                formation, while dry air maximizes atmospheric transmission.
                """
            )

    with col_exp2:
        with st.expander("📅 doy_sin & doy_cos – Seasonal Astronomy (Combined Drop ~0.151)", expanded=True):
            st.markdown(
                """
                **Why it matters:** Earth's 23.44° axial tilt causes seasonal variations in solar declination angle. 
                Chengalpattu (12.69°N) experiences solar zenith (Sun directly overhead) twice a year in April and August. 
                Encoding day-of-year into sine and cosine provides a continuous circular seasonal coordinate without 
                discontinuities at the Dec 31 / Jan 1 boundary.
                """
            )

        with st.expander("📊 T_RANGE – Diurnal Temperature Swing (Drop ~0.055)", expanded=True):
            st.markdown(
                """
                **Why it matters:** Defined as `T2M_MAX - T2M_MIN`. On cloudless days, maximum solar heating pushes 
                afternoon temperatures high, while unobstructed radiative cooling drops nighttime temperatures low 
                (large range, often 10–14°C). On cloudy or rainy days, clouds act as an insulator at night and a mirror by day, 
                shrinking the diurnal swing to 3–5°C.
                """
            )

        with st.expander("💨 WS2M – Wind Speed (Drop ~0.025)", expanded=True):
            st.markdown(
                """
                **Why it matters:** Wind advects moist maritime air from the Bay of Bengal inland toward Chengalpattu. 
                High wind episodes often accompany low-pressure depressions and monsoon squalls that bring rapid cloud 
                movement and atmospheric turbidity.
                """
            )

    st.markdown("---")

    # SHAP Summary Section
    st.subheader("3. SHAP (SHapley Additive exPlanations) Global Summary")
    st.caption(
        "SHAP values assign each feature an additive contribution toward each individual daily prediction, "
        "rooted in cooperative game theory."
    )

    if SHAP_SUMMARY_PATH.exists():
        st.image(
            str(SHAP_SUMMARY_PATH),
            caption="Figure: Pre-computed SHAP Summary Plot for Random Forest on 731 Unseen Test Days (2023–2024).",
            use_container_width=True,
        )

        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-top: 12px; font-size: 0.9rem; line-height: 1.5;">
                <b>How to read this SHAP Bee-Swarm Plot:</b><br>
                • <b>Vertical Ordering:</b> Features are ranked from most influential (top) to least influential (bottom).<br>
                • <b>Horizontal Position:</b> Each dot represents one test day. The X-axis represents the SHAP value (impact on predicted irradiance in kWh/m²/day). Points to the right push irradiance <b>higher</b>; points to the left pull irradiance <b>lower</b>.<br>
                • <b>Color:</b> <span style="color: #EF4444; font-weight: bold;">Red</span> denotes high feature values; <span style="color: #3B82F6; font-weight: bold;">Blue</span> denotes low feature values.<br>
                • <b>Physical Insight:</b> Notice how high precipitation (red dots) and high humidity (red dots) concentrate on the extreme left, dragging solar predictions down by 2.0 to 3.5 kWh/m²/day. Conversely, high temperatures and high diurnal range (red dots) consistently push predictions upward.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("SHAP plot image is being prepared.")
