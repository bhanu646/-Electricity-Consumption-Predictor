"""
Model Training and Evaluation Module.
Trains Linear Regression, Decision Tree Regressor, and Random Forest Regressor.
Computes MAE, MSE, RMSE, R2 Score, selects the best model, and persists artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.data_preprocessing import get_preprocessor, prepare_data_and_split, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_FILE_PATH = os.path.join(MODELS_DIR, "trained_model.pkl")
METADATA_FILE_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

def get_models_dict():
    """Returns the candidate models dictionary."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    }

def train_and_evaluate_all(df, test_size=0.2, random_state=42):
    """
    Trains all candidate models, evaluates them on the test set,
    and returns metrics, predictions, and trained pipelines.
    """
    X_train, X_test, y_train, y_test, X, y = prepare_data_and_split(
        df, test_size=test_size, random_state=random_state
    )

    models = get_models_dict()
    results = []
    pipelines = {}
    test_predictions = {}
    train_predictions = {}

    for name, model in models.items():
        # Linear regression benefits from numeric scaling; trees work fine with either
        preprocessor = get_preprocessor(scale_numeric=True)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # Train pipeline
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred_test = pipeline.predict(X_test)
        y_pred_train = pipeline.predict(X_train)

        # Compute metrics
        mae = mean_absolute_error(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_test)

        train_r2 = r2_score(y_train, y_pred_train)

        results.append({
            "Model": name,
            "MAE": round(mae, 3),
            "MSE": round(mse, 3),
            "RMSE": round(rmse, 3),
            "R2_Score": round(r2, 4),
            "Train_R2": round(train_r2, 4)
        })

        pipelines[name] = pipeline
        test_predictions[name] = y_pred_test
        train_predictions[name] = y_pred_train

    comparison_df = pd.DataFrame(results)

    # Best model selection based on highest R2 Score (or lowest RMSE)
    best_row = comparison_df.sort_values(by="R2_Score", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_pipeline = pipelines[best_model_name]

    return {
        "comparison_df": comparison_df,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "pipelines": pipelines,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "test_predictions": test_predictions,
        "train_predictions": train_predictions,
        "consumption_90th_percentile": float(np.percentile(y, 90)),
        "consumption_mean": float(np.mean(y)),
        "consumption_max": float(np.max(y)),
        "consumption_min": float(np.min(y))
    }

def get_feature_names(pipeline):
    """Extracts transformed feature names from the Scikit-learn Pipeline."""
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        return NUMERICAL_FEATURES + cat_feature_names
    except Exception:
        return NUMERICAL_FEATURES + CATEGORICAL_FEATURES

def extract_feature_importances(pipeline, model_name):
    """
    Extracts feature importances or coefficients from the trained model pipeline.
    """
    feature_names = get_feature_names(pipeline)
    regressor = pipeline.named_steps['regressor']
    
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        return df_imp
    elif hasattr(regressor, 'coef_'):
        coefs = regressor.coef_
        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': np.round(coefs, 3),
            'Absolute_Effect': np.round(np.abs(coefs), 3)
        }).sort_values(by='Absolute_Effect', ascending=False)
        return df_imp
    return None

def save_model(pipeline, model_name, metrics_dict, metadata=None,
               model_path=MODEL_FILE_PATH, metadata_path=METADATA_FILE_PATH):
    """Saves the trained pipeline and metadata to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    
    meta = {
        "best_model_name": model_name,
        "metrics": metrics_dict,
        "features": NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    }
    if metadata:
        meta.update(metadata)
        
    with open(metadata_path, 'w') as f:
        json.dump(meta, f, indent=4)

def load_saved_model(model_path=MODEL_FILE_PATH, metadata_path=METADATA_FILE_PATH):
    """Loads the saved pipeline and metadata."""
    if not os.path.exists(model_path):
        return None, None
    pipeline = joblib.load(model_path)
    meta = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
    return pipeline, meta
