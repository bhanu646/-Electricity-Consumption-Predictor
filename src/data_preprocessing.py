"""
Data Preprocessing and Feature Engineering Module.
Handles loading, data inspection, cleaning, feature engineering, and pipeline creation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

NUMERICAL_FEATURES = [
    'Temperature',
    'Humidity',
    'Number_of_People',
    'Previous_Day_Consumption',
    'Appliance_Usage',
    'Month'
]

CATEGORICAL_FEATURES = [
    'Day_of_Week',
    'Is_Weekend'
]

TARGET_COLUMN = 'Electricity_Consumption'

REQUIRED_COLUMNS = [
    'Temperature',
    'Humidity',
    'Number_of_People',
    'Previous_Day_Consumption',
    'Appliance_Usage'
]

def load_data(filepath):
    """Loads dataset from CSV file path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df

def validate_dataset(df):
    """Validates whether required columns exist in the uploaded dataframe."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    if TARGET_COLUMN not in df.columns:
        return False, f"Missing target column: '{TARGET_COLUMN}'"
    return True, "Dataset is valid."

def inspect_dataset(df):
    """Returns dataset summary metrics (shape, missing values, duplicates)."""
    summary = {
        'total_rows': int(df.shape[0]),
        'total_columns': int(df.shape[1]),
        'missing_values': int(df.isnull().sum().sum()),
        'missing_per_column': df.isnull().sum().to_dict(),
        'duplicate_rows': int(df.duplicated().sum()),
        'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    return summary

def clean_data(df):
    """Cleans dataset by removing duplicates and imputing missing values if any."""
    df_clean = df.copy()
    
    # 1. Drop duplicates
    df_clean = df_clean.drop_duplicates()
    
    # 2. Impute missing values if present
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                
    return df_clean

def engineer_features(df):
    """
    Extracts time-based features from Date column:
    - Month, Day, Day_of_Week, Is_Weekend (if not already present).
    """
    df_eng = df.copy()
    
    if 'Date' in df_eng.columns:
        df_eng['Date'] = pd.to_datetime(df_eng['Date'], errors='coerce')
        if 'Month' not in df_eng.columns:
            df_eng['Month'] = df_eng['Date'].dt.month.fillna(6).astype(int)
        if 'Day' not in df_eng.columns:
            df_eng['Day'] = df_eng['Date'].dt.day.fillna(15).astype(int)
        if 'Day_of_Week' not in df_eng.columns:
            df_eng['Day_of_Week'] = df_eng['Date'].dt.day_name().fillna('Wednesday')
        if 'Is_Weekend' not in df_eng.columns:
            df_eng['Is_Weekend'] = df_eng['Date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    else:
        # Fallback sensible defaults if Date is not provided
        if 'Month' not in df_eng.columns:
            df_eng['Month'] = 6
        if 'Day_of_Week' not in df_eng.columns:
            df_eng['Day_of_Week'] = 'Wednesday'
        if 'Is_Weekend' not in df_eng.columns:
            df_eng['Is_Weekend'] = 0
            
    # Ensure Is_Weekend is integer
    df_eng['Is_Weekend'] = df_eng['Is_Weekend'].astype(int)
    
    return df_eng

def get_preprocessor(scale_numeric=True):
    """
    Constructs a Scikit-Learn ColumnTransformer for numerical and categorical features.
    """
    if scale_numeric:
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
    else:
        numeric_transformer = 'passthrough'

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def prepare_data_and_split(df, test_size=0.2, random_state=42):
    """
    Cleans, engineers features, separates X and y, and performs train-test split.
    """
    df_clean = clean_data(df)
    df_eng = engineer_features(df_clean)
    
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df_eng[feature_cols].copy()
    y = df_eng[TARGET_COLUMN].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, X, y
