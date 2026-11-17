"""
Dashboard Page: Executive overview of the Electricity Consumption Predictor.
Displays summary KPIs, high-level distributions, model benchmark status, and quick prediction highlights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.database import get_all_predictions

def render_dashboard(df, trained_state):
    """Renders the main dashboard page."""
    
    # Hero / Header Banner
    st.markdown("""
        <div class="project-header-box">
            <h1>⚡ Electricity Consumption Predictor</h1>
            <p>
                An intelligent Machine Learning system designed to analyze household and commercial electricity usage patterns,
                uncover environmental driving factors, and predict daily consumption (kWh) with high regression accuracy.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate dataset stats
    total_records = len(df)
    avg_consumption = round(df['Electricity_Consumption'].mean(), 2)
    max_consumption = round(df['Electricity_Consumption'].max(), 2)
    min_consumption = round(df['Electricity_Consumption'].min(), 2)
    
    best_model_name = "Not Trained Yet"
    best_r2 = "N/A"
    if trained_state and 'best_model_name' in trained_state:
        best_model_name = trained_state['best_model_name']
        best_r2_val = trained_state['comparison_df'].loc[
            trained_state['comparison_df']['Model'] == best_model_name, 'R2_Score'
        ].values
        if len(best_r2_val) > 0:
            best_r2 = f"{best_r2_val[0]:.4f}"

    # KPI Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Dataset Records</div>
                <div class="metric-value">{total_records:,}</div>
                <div class="metric-subtitle">Daily observations</div>
            </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Consumption</div>
                <div class="metric-value">{avg_consumption} <span style="font-size:1rem;color:#64748b;">kWh</span></div>
                <div class="metric-subtitle">Daily baseline load</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Peak Usage</div>
                <div class="metric-value" style="color:#ef4444;">{max_consumption} <span style="font-size:1rem;color:#64748b;">kWh</span></div>
                <div class="metric-subtitle">Summer peak load</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Minimum Usage</div>
                <div class="metric-value" style="color:#10b981;">{min_consumption} <span style="font-size:1rem;color:#64748b;">kWh</span></div>
                <div class="metric-subtitle">Idle baseline load</div>
            </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Best Model R²</div>
                <div class="metric-value" style="color:#2563eb;">{best_r2}</div>
                <div class="metric-subtitle">{best_model_name.split()[0]}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2-column Quick Overview Charts
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("📈 Consumption Dynamics & Trend")
        df_sorted = df.copy()
        if 'Date' in df_sorted.columns:
            df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
            df_sorted = df_sorted.sort_values('Date')
            
            # Show last 90 days for clean, focused dashboard view
            recent_df = df_sorted.tail(90)
            fig_trend = px.line(
                recent_df,
                x='Date',
                y='Electricity_Consumption',
                title="Recent 90 Days Daily Electricity Consumption (kWh)",
                template="plotly_white",
                color_discrete_sequence=['#2563eb']
            )
            fig_trend.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=320)
            st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        st.subheader("🌡️ Temperature Impact")
        fig_scatter = px.scatter(
            df.sample(min(200, len(df)), random_state=42),
            x='Temperature',
            y='Electricity_Consumption',
            color='Appliance_Usage',
            title="Temperature vs Consumption Sample",
            template="plotly_white",
            labels={'Electricity_Consumption': 'kWh', 'Temperature': 'Temp (°C)'}
        )
        fig_scatter.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=320)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Recent Predictions Section from SQLite
    st.subheader("🕒 Recent Predictions History")
    recent_preds = get_all_predictions(limit=5)
    
    if recent_preds.empty:
        st.info("No predictions made yet. Navigate to the **'Predict Consumption'** page from the sidebar to make your first live prediction!")
    else:
        display_cols = [
            'timestamp', 'temperature', 'humidity', 'number_of_people',
            'previous_day_consumption', 'appliance_usage', 'predicted_consumption', 'model_used'
        ]
        available_cols = [c for c in display_cols if c in recent_preds.columns]
        
        # Rename for clean presentation
        col_rename = {
            'timestamp': 'Timestamp',
            'temperature': 'Temp (°C)',
            'humidity': 'Humidity (%)',
            'number_of_people': 'Occupants',
            'previous_day_consumption': 'Prev Day (kWh)',
            'appliance_usage': 'Appliances (hrs)',
            'predicted_consumption': 'Predicted (kWh)',
            'model_used': 'Model'
        }
        st.dataframe(
            recent_preds[available_cols].rename(columns=col_rename),
            use_container_width=True,
            hide_index=True
        )

    # Quick Workflow Guide
    st.markdown("---")
    st.markdown("### 🚀 Recommended Viva Walkthrough Workflow")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown("**1. Dataset Explorer**\n\nInspect data distributions, handle missing values, and review engineered features.")
    with g2:
        st.markdown("**2. Data Visualization**\n\nExplore interactive charts showing temporal trends and thermal load relationships.")
    with g3:
        st.markdown("**3. Model Training**\n\nTrain Linear Regression, Decision Tree, and Random Forest models and compare MAE, RMSE, R².")
    with g4:
        st.markdown("**4. Predict Consumption**\n\nTest real-world what-if scenarios with live predictions and warning triggers.")
