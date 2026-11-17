"""
Dataset Explorer Page: Inspect data, review statistics, explore feature engineering,
and upload custom datasets.
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.data_preprocessing import inspect_dataset, validate_dataset, engineer_features

def render_dataset_page(df, set_active_df_callback=None):
    """Renders the Dataset Explorer and CSV upload section."""
    st.title("📂 Dataset Explorer & Preprocessing")
    st.markdown("""
        Explore the historical electricity usage dataset used to train our regression models.
        Inspect summary statistics, missing values, feature distributions, or upload a custom CSV dataset.
    """)

    # 1. Dataset Overview & Key Metrics
    summary = inspect_dataset(df)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Records", f"{summary['total_rows']:,}")
    with m2:
        st.metric("Total Features", f"{summary['total_columns']}")
    with m3:
        st.metric("Missing Values", f"{summary['missing_values']}")
    with m4:
        st.metric("Duplicate Rows", f"{summary['duplicate_rows']}")

    st.markdown("---")

    # 2. Interactive Data Preview
    st.subheader("📋 Dataset Preview")
    rows_to_show = st.slider("Number of sample rows to display:", min_value=5, max_value=50, value=10, step=5)
    st.dataframe(df.head(rows_to_show), use_container_width=True)

    # 3. Descriptive Statistics
    st.subheader("📊 Descriptive Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

    # 4. Feature Engineering Explanation (Academic Requirement)
    with st.expander("🔍 Feature Engineering & Physical Explanations (Click to Expand)", expanded=True):
        st.markdown("""
        Understanding **why** each feature affects electricity consumption is crucial for explaining the project in a viva:

        * **Temperature (°C):** The strongest non-linear driver. When ambient temperatures rise above ~24°C, residential air conditioners operate at high duty cycles. Similarly, in colder months below 18°C, electric heating elements and geysers spike consumption.
        * **Humidity (%):** High humidity impairs human evaporative cooling, prompting users to set AC thermostats lower and run dehumidifiers, significantly increasing compressor power draw.
        * **Number of People:** More occupants directly increase daily domestic activities—lighting multiple rooms, charging devices, watching television, running laptops, and cooking.
        * **Previous Day Consumption (kWh):** Represents household behavioural habits and operational momentum (autoregressive nature). If heavy machinery or AC ran yesterday, there is high likelihood of similar usage patterns today.
        * **Appliance Usage (Hours):** Direct cumulative electrical load from washing machines, microwaves, dishwashers, and computers ($kWh = kW \times Hours$).
        * **Day of Week & Weekend:** People spend significantly more time at home on Saturdays and Sundays, shifting peak daytime energy consumption compared to workdays.
        * **Month & Day (from Date):** Captures seasonal climate transitions (Summer, Monsoon, Winter) and calendar effects.
        """)

    st.markdown("---")

    # 5. CSV Upload Feature
    st.subheader("📤 Upload Your Dataset")
    st.write("You can test or retrain the models with your own CSV data. Ensure your CSV matches the required columns.")

    col_up, col_down = st.columns([3, 1])

    with col_up:
        uploaded_file = st.file_uploader("Upload a CSV file for training:", type=["csv"])

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                is_valid, message = validate_dataset(uploaded_df)

                if is_valid:
                    st.success(f"✅ File uploaded successfully! {len(uploaded_df)} records detected.")
                    st.write("Preview of uploaded dataset:")
                    st.dataframe(uploaded_df.head(5), use_container_width=True)

                    if st.button("Use this Dataset for Model Training"):
                        if set_active_df_callback:
                            set_active_df_callback(uploaded_df)
                        st.session_state['active_df'] = uploaded_df
                        st.session_state['models_trained'] = False  # prompt retrain
                        st.success("🎉 Active dataset updated! Please navigate to **'Model Training'** to train the models on your new data.")
                else:
                    st.error(f"❌ Validation Error: {message}")
                    st.info("Required columns: Temperature, Humidity, Number_of_People, Previous_Day_Consumption, Appliance_Usage, Electricity_Consumption")
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

    with col_down:
        st.write("**Download Sample**")
        st.write("Download the default sample dataset for reference or testing:")
        sample_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_csv,
            file_name="electricity_consumption_sample.csv",
            mime="text/csv"
        )
