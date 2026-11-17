"""
Smoke test script to verify end-to-end ML training, evaluation, saving, and prediction.
"""

import os
import pandas as pd
from src.data_preprocessing import load_data, inspect_dataset
from src.train_model import train_and_evaluate_all, save_model, load_saved_model
from src.prediction import predict_consumption
from src.database import save_prediction, get_all_predictions

def test_pipeline():
    print("1. Loading dataset...")
    df = load_data("data/electricity_consumption.csv")
    print(f"   Loaded {len(df)} rows. Columns: {list(df.columns)}")
    
    print("\n2. Training all regression models...")
    res = train_and_evaluate_all(df)
    print("\nModel Comparison Table:")
    print(res['comparison_df'].to_string(index=False))
    print(f"\nBest Model Identified: {res['best_model_name']}")
    
    print("\n3. Saving model artifacts...")
    save_model(
        pipeline=res['best_pipeline'],
        model_name=res['best_model_name'],
        metrics_dict=res['comparison_df'].to_dict(orient='records'),
        metadata={"consumption_90th_percentile": res['consumption_90th_percentile']}
    )
    print("   Saved successfully!")
    
    print("\n4. Loading saved model...")
    loaded_pipe, meta = load_saved_model()
    assert loaded_pipe is not None, "Failed to load model!"
    print(f"   Successfully loaded {meta['best_model_name']}")
    
    print("\n5. Running test prediction...")
    test_input = {
        'Temperature': 34.5,
        'Humidity': 70.0,
        'Number_of_People': 4,
        'Previous_Day_Consumption': 32.0,
        'Day_of_Week': 'Saturday',
        'Is_Weekend': 1,
        'Appliance_Usage': 7.5,
        'Month': 6
    }
    pred_res = predict_consumption(loaded_pipe, test_input)
    print(f"   Predicted: {pred_res['predicted_value']} {pred_res['unit']}")
    print(f"   Status: {pred_res['status']}")
    print(f"   Interpretation: {pred_res['interpretation'].encode('ascii', 'replace').decode()}")
    
    print("\n6. Testing SQLite database logging...")
    save_prediction(
        temperature=test_input['Temperature'],
        humidity=test_input['Humidity'],
        number_of_people=test_input['Number_of_People'],
        previous_day_consumption=test_input['Previous_Day_Consumption'],
        day_of_week=test_input['Day_of_Week'],
        is_weekend=test_input['Is_Weekend'],
        appliance_usage=test_input['Appliance_Usage'],
        predicted_consumption=pred_res['predicted_value'],
        model_used=meta['best_model_name']
    )
    history = get_all_predictions()
    print(f"   Retrieved {len(history)} record(s) from SQLite database.")
    print("   All tests passed successfully!")

if __name__ == '__main__':
    test_pipeline()
