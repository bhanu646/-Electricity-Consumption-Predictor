"""
Visualization Page: Interactive exploratory data analysis charts.
Renders Chart 1 (Temporal Trend), Chart 2 (Temperature vs Consumption),
Chart 3 (Day of Week Average), and Correlation Matrix.
"""

import streamlit as st
import pandas as pd
from src.visualization import (
    plot_consumption_over_time,
    plot_temp_vs_consumption,
    plot_consumption_by_day,
    plot_correlation_heatmap
)

def render_visualization_page(df):
    """Renders the Data Visualization page."""
    st.title("📊 Data Visualization & Exploratory Analysis")
    st.markdown("""
        Visualizing data patterns helps uncover the physical dynamics governing power usage
        before feeding features into Machine Learning algorithms.
    """)

    # Tabbed layout for intuitive browsing
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Chart 1: Consumption Over Time",
        "🌡️ Chart 2: Temp vs. Consumption",
        "📅 Chart 3: Consumption by Day",
        "🔥 Correlation Heatmap"
    ])

    with tab1:
        st.subheader("Chart 1: Electricity Consumption Over Time")
        st.write("Examine daily electricity consumption dynamics alongside a 30-day moving average trendline.")
        
        # Date range filter if Date exists
        if 'Date' in df.columns:
            df['Date_dt'] = pd.to_datetime(df['Date'])
            min_date = df['Date_dt'].min().date()
            max_date = df['Date_dt'].max().date()
            
            c_date1, c_date2 = st.columns(2)
            with c_date1:
                start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
            with c_date2:
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
                
            filtered_df = df[(df['Date_dt'].dt.date >= start_date) & (df['Date_dt'].dt.date <= end_date)].drop(columns=['Date_dt'])
        else:
            filtered_df = df
            
        fig_time = plot_consumption_over_time(filtered_df)
        st.plotly_chart(fig_time, use_container_width=True)
        st.caption("Notice the clear summer peak where cooling demand spikes electricity usage.")

    with tab2:
        st.subheader("Chart 2: Temperature vs. Electricity Consumption")
        st.write("Analyze the non-linear relationship between outdoor ambient temperature and power demand.")
        fig_temp = plot_temp_vs_consumption(df)
        st.plotly_chart(fig_temp, use_container_width=True)
        st.info("""
            **Observation:** Above 25°C, consumption rises sharply (exponential-like slope) due to air conditioner compressors.
            The color indicates humidity (higher humidity increases AC load for dehumidification).
        """)

    with tab3:
        st.subheader("Chart 3: Average Consumption by Day of Week")
        st.write("Compare baseline weekday consumption vs weekend residential occupancy surges.")
        fig_day = plot_consumption_by_day(df)
        st.plotly_chart(fig_day, use_container_width=True)
        st.caption("Error bars represent one standard deviation across all observations for each weekday.")

    with tab4:
        st.subheader("Feature Correlation Analysis")
        st.write("Pearson correlation coefficients between all numerical features and the target variable.")
        fig_corr = plot_correlation_heatmap(df)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Positive correlation values indicate features that drive higher electricity consumption.")
