"""
Dataset Generator for Electricity Consumption Predictor.
Generates realistic, physically-sound synthetic data for academic and demonstration purposes.
"""

import os
import numpy as np
import pandas as pd

def generate_electricity_dataset(start_date="2023-01-01", end_date="2024-12-31", random_seed=42):
    np.random.seed(random_seed)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    # Day of year for seasonal cycle simulation (1 to 365/366)
    day_of_year = date_range.dayofyear.values
    
    # 1. Temperature: Seasonal variation (Hot in summer around day 140-180, cool in winter)
    # Average ~28°C, swinging between 14°C (winter) and 42°C (peak summer)
    seasonal_temp = 28 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    temp_noise = np.random.normal(0, 2.5, n_days)
    temperature = np.round(np.clip(seasonal_temp + temp_noise, 10.0, 45.0), 1)
    
    # 2. Humidity: Monsoon/seasonal variation (Higher in July-August, lower in spring)
    seasonal_humidity = 60 + 20 * np.sin(2 * np.pi * (day_of_year - 150) / 365.25)
    humidity_noise = np.random.normal(0, 6.0, n_days)
    humidity = np.round(np.clip(seasonal_humidity + humidity_noise, 25.0, 95.0), 1)
    
    # 3. Number of People: typically 2 to 6, fluctuating on weekends / holidays
    base_people = np.random.choice([2, 3, 4, 5, 6], size=n_days, p=[0.15, 0.35, 0.30, 0.15, 0.05])
    
    # 4. Day of Week & Weekend
    day_of_week = date_range.day_name()
    is_weekend = np.where(date_range.dayofweek >= 5, 1, 0)
    
    # Additional visitors on weekends occasionally
    number_of_people = base_people + np.random.binomial(1, 0.25, n_days) * is_weekend
    
    # 5. Appliance Usage (Hours per day: TV, PC, Kitchen, Washing Machine, etc.)
    # More on weekends and hotter days
    appliance_usage = np.round(
        3.5 + 0.8 * is_weekend + 0.6 * (number_of_people - 2) + 
        np.maximum(0, (temperature - 30) * 0.15) + 
        np.random.normal(0, 0.8, n_days),
        1
    )
    appliance_usage = np.clip(appliance_usage, 2.0, 14.0)
    
    # 6. Sequential Generation for Previous_Day_Consumption and Current Day Electricity_Consumption
    electricity_consumption = np.zeros(n_days)
    previous_day_consumption = np.zeros(n_days)
    
    # Initial state
    prev_val = 22.0
    
    for i in range(n_days):
        previous_day_consumption[i] = np.round(prev_val, 2)
        
        # Physical components:
        # Base household load (lighting, refrigeration, standby)
        base_load = 6.5
        
        # Occupancy load (~2.2 kWh per person)
        occupancy_load = 2.2 * number_of_people[i]
        
        # Appliance load (~1.4 kWh per hour of usage)
        appliance_load = 1.35 * appliance_usage[i]
        
        # Cooling / Heating Load based on thermal comfort (24°C balance point)
        temp = temperature[i]
        if temp > 24.0:
            cooling_load = 0.75 * ((temp - 24.0) ** 1.25)
            # Extra cooling load if humidity is high (latent heat)
            if humidity[i] > 60.0:
                cooling_load += 0.04 * (humidity[i] - 60.0)
        elif temp < 18.0:
            cooling_load = 0.50 * ((18.0 - temp) ** 1.1)
        else:
            cooling_load = 0.0
            
        # Weekend staying home bump
        weekend_load = 2.0 if is_weekend[i] == 1 else 0.0
        
        # Inertia from previous day (habits carry over)
        inertia_load = 0.15 * (prev_val - 22.0)
        
        # Random noise
        noise = np.random.normal(0, 1.2)
        
        total = base_load + occupancy_load + appliance_load + cooling_load + weekend_load + inertia_load + noise
        consumption = max(8.5, round(total, 2))
        
        electricity_consumption[i] = consumption
        prev_val = consumption

    df = pd.DataFrame({
        'Date': date_range.strftime('%Y-%m-%d'),
        'Temperature': temperature,
        'Humidity': humidity,
        'Number_of_People': number_of_people,
        'Previous_Day_Consumption': previous_day_consumption,
        'Day_of_Week': day_of_week,
        'Is_Weekend': is_weekend,
        'Appliance_Usage': appliance_usage,
        'Electricity_Consumption': electricity_consumption
    })
    
    return df

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(data_dir)
    
    df = generate_electricity_dataset()
    
    # Save in both data/ and project root for convenience and user requirement compatibility
    data_csv_path = os.path.join(data_dir, 'electricity_consumption.csv')
    root_csv_path = os.path.join(project_root, 'electricity_consumption.csv')
    
    df.to_csv(data_csv_path, index=False)
    df.to_csv(root_csv_path, index=False)
    
    print(f"Dataset generated successfully with {len(df)} records.")
    print(f"Saved to: {data_csv_path}")
    print(f"Saved to: {root_csv_path}")
    print("\nDataset Summary:")
    print(df.describe())
