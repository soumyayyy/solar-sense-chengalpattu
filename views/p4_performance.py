"""
Page 4: Model Performance
Presents benchmark metrics, comparative bar charts, predicted-vs-actual scatter plots,
time-series drilldowns, monthly error patterns, and the time-based validation strategy.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.models import load_metrics_df, load_test_predictions_df


def render():
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; color: #1E293B;">📊 Model Performance & Benchmarking</h2>
            <p style="color: #64748B; margin-top: 4px; font-size: 0.95rem;">
                Rigorous evaluation on unseen 2023–2024 test data (731 days) across regression metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics_df = load_metrics_df()
    test_preds_df = load_test_predictions_df()

    # Metrics Table with Highlights
    st.subheader("1. Comprehensive Metrics Table")

    # Format dataframe for display
    display_df = metrics_df.copy()
    display_df["MAE"] = display_df["MAE"].map(lambda x: f"{x:.3f}")
    display_df["MSE"] = display_df["MSE"].map(lambda x: f"{x:.3f}")
    display_df["RMSE"] = display_df["RMSE"].map(lambda x: f"{x:.3f}")
    display_df["R2"] = display_df["R2"].map(lambda x: f"{x:.3f}")

    col_tbl, col_summary = st.columns([3, 2])
    with col_tbl:
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with col_summary:
        st.markdown(
            """
            <div style="background: #F1F5F9; border-radius: 8px; padding: 14px; font-size: 0.88rem; line-height: 1.45;">
                <b>Key Observations:</b><br>
                • <b>SVR</b> achieved the lowest <b>MAE (0.560 kWh/m²/day)</b>, outperforming the climatology baseline by <b>28.9%</b>.<br>
                • <b>Random Forest</b> achieved the highest <b>R² (0.686)</b> and lowest <b>RMSE (0.749)</b>, closely tied with SVR.<br>
                • Both nonlinear ML models significantly outperform <b>Linear Regression (MAE 0.673, R² 0.577)</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Comparative Metric Bar Charts
    st.subheader("2. Metric Comparison Across Models")
    m_tabs = st.tabs(["MAE (Mean Absolute Error)", "RMSE (Root Mean Squared Error)", "R² Score (Explained Variance)"])

    models_order = metrics_df["Model"].tolist()
    palette = ["#94A3B8", "#FB923C", "#2563EB", "#059669"]

    with m_tabs[0]:
        fig_mae = go.Figure(
            data=[
                go.Bar(
                    x=metrics_df["Model"],
                    y=metrics_df["MAE"],
                    marker=dict(color=palette),
                    text=[f"{v:.3f}" for v in metrics_df["MAE"]],
                    textposition="outside",
                )
            ]
        )
        fig_mae.update_layout(
            title="MAE by Model (Lower is Better)",
            yaxis_title="MAE (kWh/m²/day)",
            template="plotly_white",
            height=340,
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    with m_tabs[1]:
        fig_rmse = go.Figure(
            data=[
                go.Bar(
                    x=metrics_df["Model"],
                    y=metrics_df["RMSE"],
                    marker=dict(color=palette),
                    text=[f"{v:.3f}" for v in metrics_df["RMSE"]],
                    textposition="outside",
                )
            ]
        )
        fig_rmse.update_layout(
            title="RMSE by Model (Penalizes Large Errors; Lower is Better)",
            yaxis_title="RMSE (kWh/m²/day)",
            template="plotly_white",
            height=340,
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    with m_tabs[2]:
        fig_r2 = go.Figure(
            data=[
                go.Bar(
                    x=metrics_df["Model"],
                    y=metrics_df["R2"],
                    marker=dict(color=palette),
                    text=[f"{v:.3f}" for v in metrics_df["R2"]],
                    textposition="outside",
                )
            ]
        )
        fig_r2.update_layout(
            title="R² Score by Model (Higher is Better)",
            yaxis_title="R² (Coefficient of Determination)",
            template="plotly_white",
            height=340,
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("---")

    # Predicted vs Actual Scatter Plot
    st.subheader("3. Predicted vs Actual Scatter Analysis")
    selected_model = st.selectbox(
        "Choose Model for Scatter Inspection",
        options=["SVR", "Random Forest", "Linear Regression", "Baseline (climatology)"],
        index=0,
    )

    actual = test_preds_df["ALLSKY_SFC_SW_DWN"]
    pred = test_preds_df[selected_model]

    fig_scatter = go.Figure()

    # Scatter points
    fig_scatter.add_trace(
        go.Scatter(
            x=actual,
            y=pred,
            mode="markers",
            marker=dict(
                size=6,
                color="#2563EB",
                opacity=0.5,
                line=dict(width=0.5, color="#1E293B"),
            ),
            name="Test Days (2023–2024)",
        )
    )

    # 1:1 Reference Line
    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    fig_scatter.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="#EF4444", dash="dash", width=2),
            name="Ideal 1:1 Parity",
        )
    )

    fig_scatter.update_layout(
        title=f"{selected_model}: Actual vs Predicted Irradiance (Test Set)",
        xaxis_title="Actual NASA Irradiance (kWh/m²/day)",
        yaxis_title="Predicted Irradiance (kWh/m²/day)",
        template="plotly_white",
        height=450,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # Mean Absolute Error by Month
    st.subheader("4. Error Distribution by Calendar Month")
    st.caption(
        "Atmospheric cloud dynamics vary strongly between dry summer months and the wet Northeast Monsoon (October–December) in Tamil Nadu."
    )

    test_preds_df["Month"] = test_preds_df.index.month
    test_preds_df["Month_Name"] = test_preds_df.index.strftime("%b")
    test_preds_df["Abs_Error"] = np.abs(actual - pred)

    monthly_mae = (
        test_preds_df.groupby("Month")
        .agg({"Abs_Error": "mean", "Month_Name": "first"})
        .sort_index()
    )

    fig_month = go.Figure(
        data=[
            go.Bar(
                x=monthly_mae["Month_Name"],
                y=monthly_mae["Abs_Error"],
                marker=dict(
                    color=[
                        "#EF4444" if m in [10, 11, 12] else "#F59E0B"
                        for m in monthly_mae.index
                    ]
                ),
                text=[f"{v:.2f}" for v in monthly_mae["Abs_Error"]],
                textposition="outside",
            )
        ]
    )
    fig_month.update_layout(
        title=f"{selected_model}: Mean Absolute Error by Month (Red = Northeast Monsoon Period)",
        xaxis_title="Month",
        yaxis_title="Mean Absolute Error (kWh/m²/day)",
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig_month, use_container_width=True)

    # Train / Test Split Strategy Explanation
    st.markdown("---")
    st.subheader("5. Validation Strategy: Time-Based Split (No Temporal Leakage)")
    st.markdown(
        """
        <div style="background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 10px; padding: 18px;">
            <h4 style="margin-top: 0; color: #1E293B;">🛡️ Avoiding Data Leakage in Environmental Time Series</h4>
            <p style="font-size: 0.92rem; color: #334155; line-height: 1.55;">
                In atmospheric and renewable energy modeling, standard randomized <em>k</em>-fold cross-validation is 
                <b>strictly flawed</b>. Weather states possess strong temporal autocorrelation; randomly shuffling days allows 
                the model to train on day <i>t+1</i> to predict day <i>t</i>, leading to severely over-optimistic performance 
                metrics that collapse in production.
            </p>
            <ul style="font-size: 0.92rem; color: #334155; line-height: 1.55;">
                <li><b>Strict Time-Based Split:</b> The dataset is chronologically partitioned into a <b>Training Set (2015–2022, 8 years, 2,922 days)</b> and an unseen <b>Test Set (2023–2024, 2 years, 731 days)</b>.</li>
                <li><b>Hyperparameter Tuning:</b> Within the 2015–2022 training partition, models were tuned using <code>TimeSeriesSplit(n_splits=5)</code>, where each fold only trains on past data and validates on future data.</li>
                <li><b>Final Holdout Evaluation:</b> The test set was touched only once after model freezing, guaranteeing honest out-of-sample generalization metrics.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
