"""
Prediction Module.
Formats user input into model-ready dataframes, produces predictions,
and generates human-readable interpretations and alerts.
"""

import pandas as pd
import numpy as np

def predict_consumption(pipeline, input_data, high_threshold=45.0, average_consumption=32.5):
    """
    Takes a trained pipeline and a dictionary of inputs,
    returns the predicted consumption (kWh), interpretation, and alert flags.
    
    Expected keys in input_data:
        - Temperature (float)
        - Humidity (float)
        - Number_of_People (int)
        - Previous_Day_Consumption (float)
        - Day_of_Week (str)
        - Is_Weekend (int: 0 or 1)
        - Appliance_Usage (float)
        - Month (int, optional, defaults to 6)
    """
    # Build single-row DataFrame
    row = {
        'Temperature': float(input_data['Temperature']),
        'Humidity': float(input_data['Humidity']),
        'Number_of_People': int(input_data['Number_of_People']),
        'Previous_Day_Consumption': float(input_data['Previous_Day_Consumption']),
        'Appliance_Usage': float(input_data['Appliance_Usage']),
        'Month': int(input_data.get('Month', 6)),
        'Day_of_Week': str(input_data['Day_of_Week']),
        'Is_Weekend': int(input_data['Is_Weekend'])
    }
    
    df_input = pd.DataFrame([row])
    
    # Run prediction through the pipeline
    prediction_raw = pipeline.predict(df_input)[0]
    predicted_val = round(float(prediction_raw), 2)
    
    # Ensure physical realism (non-negative)
    predicted_val = max(1.0, predicted_val)
    
    # Generate simple, student-friendly interpretation
    diff_from_avg = round(predicted_val - average_consumption, 2)
    pct_from_avg = round((diff_from_avg / average_consumption) * 100, 1)
    
    is_high = predicted_val >= high_threshold
    
    if is_high:
        status = "High Consumption Warning"
        alert_type = "warning"
        message = (
            f"⚠️ High consumption predicted ({predicted_val} kWh). "
            f"This is approximately {abs(pct_from_avg)}% above the baseline average ({average_consumption} kWh). "
            f"Likely driven by extreme ambient temperatures requiring AC/heating or heavy appliance usage."
        )
    elif predicted_val < (average_consumption * 0.75):
        status = "Low / Energy-Efficient Consumption"
        alert_type = "success"
        message = (
            f"🌱 Low consumption predicted ({predicted_val} kWh). "
            f"This is approximately {abs(pct_from_avg)}% below the baseline average. "
            f"Optimal temperature and moderate appliance usage keep power demand low."
        )
    else:
        status = "Normal Consumption"
        alert_type = "info"
        message = (
            f"⚡ Normal consumption predicted ({predicted_val} kWh). "
            f"This aligns closely with typical daily household requirements."
        )
        
    return {
        "predicted_value": predicted_val,
        "unit": "kWh",
        "status": status,
        "alert_type": alert_type,
        "interpretation": message,
        "input_features": row
    }
