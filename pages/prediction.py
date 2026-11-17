"""
Prediction Page: Live interactive electricity consumption estimation interface.
Takes user input features, generates predictions using the trained ML pipeline,
triggers alerts for unusually high consumption, and logs records to SQLite.
"""

import streamlit as st
import pandas as pd
from src.prediction import predict_consumption
from src.database import save_prediction
from src.train_model import load_saved_model

def render_prediction_page(trained_state):
    """Renders the Predict Consumption page."""
    st.title("⚡ Predict Electricity Consumption")
    st.markdown("""
        Estimate expected daily electrical energy demand (in **kWh**) by specifying
        environmental factors, occupancy, appliance runtime, and historical consumption.
    """)

    # 1. Model Availability Check
    pipeline = None
    model_name = "Unknown"
    models_available = {}

    if trained_state and 'pipelines' in trained_state:
        models_available = trained_state['pipelines']
        best_model_name = trained_state['best_model_name']
    else:
        # Try loading saved model from disk
        saved_pipeline, metadata = load_saved_model()
        if saved_pipeline is not None:
            best_model_name = metadata.get("best_model_name", "Loaded Model") if metadata else "Saved Model"
            models_available[best_model_name] = saved_pipeline
        else:
            st.warning("⚠️ No trained model found! Please go to the **'Model Training'** page to train the models first.")
            return

    # Model selector (allows comparing predictions between models!)
    st.subheader("1. Select Prediction Model")
    selected_model_name = st.selectbox(
        "Choose Model to use for this prediction:",
        options=list(models_available.keys()),
        index=0 if best_model_name not in models_available else list(models_available.keys()).index(best_model_name)
    )
    pipeline = models_available[selected_model_name]

    st.markdown("---")

    # 2. Input Form
    st.subheader("2. Enter Scenario Features")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 🌤️ Environmental & Calendar Features")
            temperature = st.slider(
                "Temperature (°C):",
                min_value=10.0,
                max_value=48.0,
                value=32.0,
                step=0.5,
                help="Ambient outdoor temperature in degrees Celsius."
            )
            
            humidity = st.slider(
                "Relative Humidity (%):",
                min_value=20.0,
                max_value=100.0,
                value=65.0,
                step=1.0,
                help="Relative outdoor humidity percentage."
            )
            
            day_of_week = st.selectbox(
                "Day of Week:",
                options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                index=5,
                help="Day of the week (Saturday and Sunday are weekends)."
            )
            
            # Auto-suggest weekend based on day selection
            auto_weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0
            is_weekend = st.radio(
                "Weekend Status:",
                options=[0, 1],
                index=auto_weekend,
                format_func=lambda x: "Weekend (Saturday/Sunday)" if x == 1 else "Weekday (Monday-Friday)",
                horizontal=True,
                help="Specifies whether the day falls on a weekend."
            )

        with col2:
            st.markdown("##### 🏠 Household & Operational Features")
            number_of_people = st.slider(
                "Number of Occupants / People:",
                min_value=1,
                max_value=10,
                value=4,
                step=1,
                help="Total number of residents or active occupants in the premises."
            )

            previous_day_consumption = st.number_input(
                "Previous Day Consumption (kWh):",
                min_value=5.0,
                max_value=100.0,
                value=30.0,
                step=0.5,
                help="Yesterday's total electricity usage in kilowatt-hours."
            )

            appliance_usage = st.slider(
                "Major Appliance Usage (Hours/day):",
                min_value=1.0,
                max_value=16.0,
                value=6.0,
                step=0.5,
                help="Combined cumulative operating hours of major appliances (e.g., AC, TV, washing machine, dishwasher)."
            )

            month = st.selectbox(
                "Month of Year:",
                options=list(range(1, 13)),
                index=5,  # June
                format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x - 1],
                help="Month helps capture broader seasonal weather patterns."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.form_submit_button("⚡ Predict Consumption", type="primary", use_container_width=True)

    # 3. Handle Prediction Submission
    if predict_btn:
        high_threshold = 45.0
        if trained_state and 'consumption_90th_percentile' in trained_state:
            high_threshold = trained_state['consumption_90th_percentile']

        input_data = {
            'Temperature': temperature,
            'Humidity': humidity,
            'Number_of_People': number_of_people,
            'Previous_Day_Consumption': previous_day_consumption,
            'Day_of_Week': day_of_week,
            'Is_Weekend': is_weekend,
            'Appliance_Usage': appliance_usage,
            'Month': month
        }

        with st.spinner("Calculating ML prediction..."):
            result = predict_consumption(
                pipeline=pipeline,
                input_data=input_data,
                high_threshold=high_threshold
            )

        pred_val = result['predicted_value']
        
        # Display Results
        st.markdown("### 🎯 Prediction Results")
        
        # Big display card
        st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 1.1rem; color: #475569; font-weight: 600;">ESTIMATED ELECTRICITY USAGE</div>
                <div class="prediction-val">{pred_val:.2f} <span class="prediction-unit">kWh</span></div>
                <div style="font-size: 0.95rem; color: #64748b;">
                    Model Used: <span class="badge badge-blue">{selected_model_name}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Interpretation and Alerts
        if result['alert_type'] == 'warning':
            st.warning(result['interpretation'])
            st.error("🚨 **High consumption predicted.** Consider optimizing air conditioner setpoints or deferring non-essential appliance loads.")
        elif result['alert_type'] == 'success':
            st.success(result['interpretation'])
        else:
            st.info(result['interpretation'])

        # Contextual metric comparisons
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            baseline_avg = 32.54
            diff = round(pred_val - baseline_avg, 2)
            st.metric(
                label="Difference from Dataset Baseline (32.54 kWh)",
                value=f"{pred_val:.2f} kWh",
                delta=f"{diff:+.2f} kWh",
                delta_color="inverse"
            )
        with c_m2:
            # Approximate domestic cost estimate (for student project context)
            unit_rate = 7.50  # Average domestic power tariff in currency units/kWh
            est_cost = round(pred_val * unit_rate, 2)
            st.metric(
                label="Estimated Daily Cost (at ~₹7.50 / unit)",
                value=f"₹{est_cost:,.2f}",
                delta=f"Based on {pred_val} units"
            )

        # 4. Save Prediction into SQLite Database
        try:
            save_prediction(
                temperature=temperature,
                humidity=humidity,
                number_of_people=number_of_people,
                previous_day_consumption=previous_day_consumption,
                day_of_week=day_of_week,
                is_weekend=is_weekend,
                appliance_usage=appliance_usage,
                predicted_consumption=pred_val,
                model_used=selected_model_name
            )
            st.toast("Record saved to prediction history!", icon="💾")
        except Exception as e:
            st.error(f"Could not log prediction to database: {str(e)}")
