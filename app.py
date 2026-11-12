"""
Solar Sense – Chengalpattu
Streamlit Web Application Entrypoint
Course: 21CEO310T – Global Warming & Climate Change
"""

import streamlit as st

# Configure wide page layout and branding
st.set_page_config(
    page_title="Solar Sense – Chengalpattu",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Solar Aesthetic
st.markdown(
    """
    <style>
    /* Global typography & card aesthetic */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Highlight metrics cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
        color: #B45309;
    }
    
    /* Primary buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Clean dataframe styling */
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
    }
    
    /* Sidebar header styling */
    .sidebar-course-badge {
        background: #FEF3C7;
        color: #92400E;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Import view modules
from views import (
    p1_overview,
    p2_live,
    p3_simulator,
    p4_performance,
    p5_explainability,
    p6_calculator,
    p7_about,
)
from config import (
    COURSE_CODE,
    COURSE_NAME,
    LOCATION_NAME,
    LATITUDE,
    LONGITUDE,
    BEST_MODEL_NAME,
)

# Sidebar Navigation & Context
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding-bottom: 12px; border-bottom: 1px solid #E2E8F0; margin-bottom: 16px;">
            <div class="sidebar-course-badge">🎓 {COURSE_CODE}</div>
            <h3 style="margin: 0; color: #1E293B; font-weight: 800;">☀️ Solar Sense</h3>
            <div style="font-size: 0.85rem; color: #64748B; margin-top: 2px;">
                Chengalpattu ({LATITUDE}°N, {LONGITUDE}°E)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page_selection = st.radio(
        "Navigation Menu",
        options=[
            "🏠 Home / Overview",
            "⚡ Live Check",
            "🎛️ What-if Simulator",
            "📊 Model Performance",
            "🔍 Explainability",
            "🌱 Solar & CO₂ Calculator",
            "ℹ️ About & Data",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size: 0.82rem; color: #475569; line-height: 1.5;">
            <b>Benchmark Model:</b> {BEST_MODEL_NAME} (MAE 0.560)<br>
            <b>Data Provider:</b> NASA POWER API<br>
            <b>Target:</b> Daily Ground Irradiance<br>
            <b>Unit:</b> kWh/m²/day
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption(
        f"📘 **Course Project**: {COURSE_CODE} – {COURSE_NAME}\n\n"
        "Built with Streamlit, Scikit-learn, Plotly & NASA POWER."
    )

# Routing
if page_selection == "🏠 Home / Overview":
    p1_overview.render()
elif page_selection == "⚡ Live Check":
    p2_live.render()
elif page_selection == "🎛️ What-if Simulator":
    p3_simulator.render()
elif page_selection == "📊 Model Performance":
    p4_performance.render()
elif page_selection == "🔍 Explainability":
    p5_explainability.render()
elif page_selection == "🌱 Solar & CO₂ Calculator":
    p6_calculator.render()
elif page_selection == "ℹ️ About & Data":
    p7_about.render()
