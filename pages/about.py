"""
About Project Page: Complete academic project details, theoretical foundations,
architecture pipeline, algorithms overview, and viva reference.
"""

import streamlit as st

def render_about_page():
    """Renders the About Project page."""
    st.title("ℹ️ About the Project")
    st.markdown("""
        **Electricity Consumption Predictor** is an academic Machine Learning project developed for
        demonstrating end-to-end regression workflows, exploratory data analysis, data preprocessing,
        feature engineering, model evaluation, and deployment using Streamlit.
    """)

    # 1. Project Specifications Grid
    st.subheader("📌 Project Profile & Metadata")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        * **Project Title:** Electricity Consumption Predictor
        * **Academic Level:** 3rd-Year B.Tech Computer Science / AI / Data Science
        * **Domain:** Applied Machine Learning & Energy Analytics
        * **ML Paradigm:** Supervised Learning
        * **Problem Type:** Continuous Numerical Regression
        """)
    with col2:
        st.markdown("""
        * **Target Variable:** `Electricity_Consumption` (continuous in kWh)
        * **Implementation Language:** Python 3
        * **Core Frameworks:** Scikit-Learn, Pandas, NumPy, Plotly, Streamlit
        * **Persistence Layer:** Joblib (Pipelines) & SQLite (Audit History)
        """)

    st.markdown("---")

    # 2. Problem Statement & Objective
    st.subheader("🎯 Problem Statement & Objective")
    
    st.markdown("""
    ### Problem Statement
    Modern residential and commercial premises experience substantial variations in electrical energy demand.
    Consumption fluctuates widely based on weather fluctuations (extreme summer heat prompting air conditioning or winter chills requiring heating),
    occupancy levels, day of the week, cumulative major appliance runtime, and habitual previous usage.
    Accurately estimating upcoming daily electricity consumption enables users to manage costs, prevents circuit overloading,
    and supports electrical grid operators in load dispatch planning.

    ### Objective
    To build an interpretable, reproducible, web-based Machine Learning application capable of:
    1. Preprocessing multi-feature tabular energy consumption data without data leakage.
    2. Engineering date-based and behavioral features.
    3. Training, validating, and benchmarking multiple standard regression algorithms.
    4. Automatically selecting the top-performing model according to statistical regression metrics ($R^2$, RMSE, MAE).
    5. Serving real-time inference via an interactive graphical user interface with high consumption warnings.
    """)

    st.markdown("---")

    # 3. Machine Learning Algorithms Explanations
    st.subheader("🧠 Machine Learning Algorithms Implemented")

    with st.expander("1. Linear Regression (Baseline Parametric Model)", expanded=True):
        st.markdown("""
        * **Concept:** Models the relationship between dependent target variable $Y$ and independent explanatory features $X_1, X_2, \\dots, X_p$ as a linear equation:
          $$Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\dots + \\beta_p X_p + \\epsilon$$
        * **Optimization:** Finds the optimal vector of coefficients $\\beta$ that minimizes the Residual Sum of Squares (Ordinary Least Squares - OLS).
        * **Role in Project:** Serves as the transparent benchmark baseline. Fast, interpretable coefficients, but assumes linear relationships and additive effects.
        """)

    with st.expander("2. Decision Tree Regressor (Non-Parametric Tree)", expanded=True):
        st.markdown("""
        * **Concept:** Recursively partitions the feature space into rectangular sub-regions by choosing splits that maximize the reduction in variance (or Mean Squared Error):
          $$\\text{MSE}_{\\text{split}} = \\frac{1}{N} \\sum_{i \\in R_m} (y_i - \\bar{y}_{R_m})^2$$
        * **Prediction:** Assigns the average target value of training samples residing within the leaf node containing the query point.
        * **Role in Project:** Effectively captures non-linear thresholds (e.g. sharp AC turn-on above 25°C), but single trees can be prone to high variance/overfitting.
        """)

    with st.expander("3. Random Forest Regressor (Ensemble Bagging)", expanded=True):
        st.markdown("""
        * **Concept:** An ensemble meta-estimator that constructs a forest of hundreds of randomized decorrelated decision trees (Bootstrap Aggregating / Bagging).
        * **Mechanism:** Each individual tree is trained on a distinct bootstrap sample with random feature subsampling at each split node. Final prediction is the arithmetic average across all trees:
          $$\\hat{Y} = \\frac{1}{B} \\sum_{b=1}^B T_b(x)$$
        * **Role in Project:** Drastically reduces variance without increasing bias, robust to outliers, captures intricate feature interactions (temperature $\\times$ humidity $\\times$ appliances), and typically yields the highest $R^2$ score.
        """)

    st.markdown("---")

    # 4. Evaluation Metrics
    st.subheader("📏 Evaluation Metrics Explained")
    
    st.markdown("""
    | Metric | Mathematical Formula | Physical Meaning |
    | :--- | :--- | :--- |
    | **MAE (Mean Absolute Error)** | $\\frac{1}{n}\\sum |y_i - \\hat{y}_i|$ | Average linear prediction error in kWh. Robust to extreme outliers. |
    | **MSE (Mean Squared Error)** | $\\frac{1}{n}\\sum (y_i - \\hat{y}_i)^2$ | Average squared deviation. Heavily penalizes large misses. |
    | **RMSE (Root Mean Squared Error)** | $\\sqrt{\\text{MSE}}$ | Square root of MSE; brings error magnitude back to the original units (kWh). |
    | **$R^2$ (Coefficient of Determination)** | $1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2}$ | Fraction of variance in electricity usage explained by the model ($1.0$ is perfect). |
    """)

    st.markdown("---")

    # 5. Project Architecture Pipeline
    st.subheader("🏗️ System Architecture & Workflow")
    st.markdown("""
    ```
    ┌──────────────────────┐
    │  Raw Dataset (CSV)   │ (731 daily records over 2 years)
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Data Preprocessing  │ Clean duplicates, impute missing values,
    │& Feature Engineering │ extract Month, Day, Weekday, Is_Weekend
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  ColumnTransformer   │ StandardScaler for numeric,
    │     & Pipelines      │ OneHotEncoder for categoricals (Zero Data Leakage)
    └──────────┬───────────┘
               │
               ▼
    ┌────────────────────────────────────────────────────────┐
    │          Model Training & Hyperparameter Tuning        │
    │  ┌──────────────────┐ ┌───────────────┐ ┌────────────┐ │
    │  │Linear Regression │ │ Decision Tree │ │Random Forest││
    │  └──────────────────┘ └───────────────┘ └────────────┘ │
    └──────────┬─────────────────────────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Model Evaluation &   │ Benchmark MAE, MSE, RMSE, R²;
    │ Model Selection      │ Select optimal pipeline & persist with Joblib
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Streamlit Interface  │ Interactive Prediction Form, High Usage Warning,
    │  & SQLite Audit Log  │ Persistence in data/predictions.db
    └──────────────────────┘
    ```
    """)
