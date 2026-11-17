# ⚡ Electricity Consumption Predictor

An end-to-end Machine Learning web application designed to forecast daily electricity consumption (kWh) based on environmental conditions, household occupancy, historical usage, and appliance runtimes.

Developed as a beginner-to-medium level **3rd-year B.Tech academic project** using Python, Scikit-Learn, and Streamlit.

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Features](#4-features)
5. [Technologies Used](#5-technologies-used)
6. [Machine Learning Algorithms](#6-machine-learning-algorithms)
7. [Dataset Description](#7-dataset-description)
8. [Project Architecture](#8-project-architecture)
9. [Installation](#9-installation)
10. [How to Run](#10-how-to-run)
11. [How the ML Model Works](#11-how-the-ml-model-works)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Screenshots & UI Walkthrough](#13-screenshots--ui-walkthrough)
14. [Future Scope](#14-future-scope)
15. [Limitations](#15-limitations)
16. [Conclusion](#16-conclusion)

---

## 1. Introduction
Electrical energy demand fluctuates continuously across residential, commercial, and institutional facilities. Fluctuations are heavily governed by weather variations (ambient heat driving air-conditioning, winter chills driving heating), family or building occupancy levels, weekend schedules, and cumulative appliance operation.

The **Electricity Consumption Predictor** provides an accessible yet rigorous Machine Learning application that models these non-linear relationships. It demonstrates standard ML regression workflows—from dataset exploration and pipeline preprocessing to multi-model benchmarking, automated model selection, and real-time inference with persistent SQLite logging.

---

## 2. Problem Statement
Electricity consumers frequently encounter unexpectedly high utility bills due to unmonitored air conditioning, unoptimized appliance usage, or seasonal shifts. Utility providers also face challenges balancing generation loads without accurate forward projections.

This project addresses the challenge of accurately estimating daily electricity consumption ($kWh$) using readily observable predictors (temperature, humidity, occupancy count, previous day usage, and appliance hours).

---

## 3. Objectives
* Build an end-to-end Machine Learning regression solution accessible to 3rd-year engineering students.
* Preprocess and engineer tabular time-based and environmental features with zero data leakage.
* Train and benchmark three regression algorithms:
  * **Linear Regression** (Parametric Baseline)
  * **Decision Tree Regressor** (Non-linear Tree Model)
  * **Random Forest Regressor** (Ensemble Bagging)
* Compare models using regression metrics: **MAE**, **MSE**, **RMSE**, and **$R^2$ Score**.
* Deploy an interactive, modern **Streamlit web application** with live what-if predictions, warnings for peak consumption, and persistent history tracking.

---

## 4. Features
* **Executive Dashboard:** Summary metric cards showing total dataset size, average/maximum/minimum consumption, current best model, trend charts, and recent predictions.
* **Dataset Explorer:** View raw and engineered features, descriptive statistics, missing value indicators, and feature domain explanations.
* **Interactive Data Visualization:**
  * Chart 1: Electricity consumption over time with 30-day rolling averages and date range filtering.
  * Chart 2: Ambient temperature vs. electricity consumption (with humidity color scale and appliance usage bubble size).
  * Chart 3: Day-of-week consumption patterns with error bars.
  * Bonus Chart: Feature correlation heatmap.
* **Model Training & Comparison:**
  * Side-by-side benchmark table ($MAE$, $MSE$, $RMSE$, $R^2$).
  * Automated best model selection based on test set generalization.
  * Chart 4: Actual vs. Predicted scatter plot with $45^\circ$ reference line.
  * Chart 5: Model performance metric comparison bar chart.
  * Chart 6: Feature importance visualizer for tree models and linear coefficients.
* **Live Prediction Interface:**
  * Intuitive sliders and selectors for temperature, humidity, people, previous day kWh, appliance hours, and weekdays.
  * Instant inference in kilowatt-hours ($kWh$) and estimated domestic cost.
  * Automated **"High Consumption Warning"** banner if predictions exceed the 90th percentile baseline.
  * Switch between models to contrast algorithmic predictions.
* **Prediction History & Audit Log:**
  * Stores all user inferences in a local SQLite database (`data/predictions.db`).
  * Filter history by model, clear history, or export to CSV.
* **Dataset Upload & Download:**
  * Upload custom `.csv` files with validation of required schema columns.
  * Download the default sample dataset for reference.

---

## 5. Technologies Used
* **Programming Language:** Python 3 (Python 3.10+)
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Machine Learning & Preprocessing:** Scikit-Learn (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`)
* **Interactive Data Visualization:** Plotly Graph Objects & Plotly Express, Matplotlib
* **Model Serialization:** Joblib
* **Database & Persistence:** SQLite3 (standard Python library)
* **Web Framework & UI:** Streamlit

---

## 6. Machine Learning Algorithms

### 1. Linear Regression
* **Type:** Parametric linear model.
* **Formula:** $Y = \beta_0 + \sum_{j=1}^p \beta_j X_j + \epsilon$
* **Function in Project:** Serves as the fundamental interpretable baseline. Evaluates how much variance can be captured assuming purely linear and additive relationships.

### 2. Decision Tree Regressor
* **Type:** Non-parametric recursive partitioning tree.
* **Splitting Criterion:** Variance Reduction / Mean Squared Error minimization.
* **Function in Project:** Captures non-linear thresholds (such as sharp AC turn-ons once temperatures surpass 24°C).

### 3. Random Forest Regressor
* **Type:** Ensemble Bootstrap Aggregating (Bagging) model.
* **Mechanism:** Averages predictions across 100 decorrelated decision trees, each trained on a bootstrap sample of data with randomized feature subsets at each node.
* **Function in Project:** Achieves the highest performance by reducing variance and minimizing overfitting on tabular feature interactions.

---

## 7. Dataset Description
The project includes a 2-year realistic daily observation dataset (`electricity_consumption.csv`, 731 records).

| Column Name | Data Type | Description | Example Range |
| :--- | :--- | :--- | :--- |
| `Date` | Date String | Observation date (`YYYY-MM-DD`) | 2023-01-01 to 2024-12-31 |
| `Temperature` | Float | Outdoor ambient temperature in Celsius | 10.0°C – 45.0°C |
| `Humidity` | Float | Relative atmospheric humidity percentage | 25.0% – 95.0% |
| `Number_of_People` | Integer | Total household occupants / residents | 1 – 7 people |
| `Previous_Day_Consumption` | Float | Electricity used on the preceding day ($kWh$) | 11.0 – 72.0 kWh |
| `Day_of_Week` | String | Monday, Tuesday, ..., Sunday | Monday – Sunday |
| `Is_Weekend` | Integer | Binary flag (1 if Saturday/Sunday, else 0) | 0 or 1 |
| `Appliance_Usage` | Float | Operating hours of major electrical appliances | 2.0 – 14.0 hours |
| **`Electricity_Consumption`** | Float | **Target Variable:** Total daily consumption ($kWh$) | 11.0 – 72.7 kWh |

*Note on Data Source:* The sample dataset is realistically synthesized using thermodynamic degree-day equations, occupancy schedules, and autoregressive behavioral inertia for demonstration and educational purposes.

---

## 8. Project Architecture

```
electricity-consumption-predictor/
│
├── app.py                     # Main Streamlit application and page router
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation
├── VIVA.md                    # 25 Viva Voce preparation questions & answers
├── test_pipeline.py           # Backend verification smoke test script
├── electricity_consumption.csv# Dataset (root copy for convenience)
│
├── data/
│   ├── electricity_consumption.csv # Active primary dataset (731 records)
│   ├── generate_data.py            # Physics-based synthetic data generator
│   └── predictions.db              # SQLite prediction audit database
│
├── models/
│   ├── trained_model.pkl           # Serialized Scikit-Learn Pipeline
│   └── model_metadata.json         # Best model metadata and benchmark metrics
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data cleaning, feature engineering, pipeline setup
│   ├── train_model.py              # Multi-model training, benchmarking, saving/loading
│   ├── prediction.py               # Live prediction logic, interpretation, alerts
│   ├── database.py                 # SQLite database helper functions
│   └── visualization.py            # Interactive Plotly chart builders
│
├── pages/
│   ├── __init__.py
│   ├── dashboard.py                # Executive KPI dashboard view
│   ├── dataset.py                  # Dataset inspection & CSV upload view
│   ├── visualization.py            # Exploratory data analysis view
│   ├── model_training.py           # Model training, comparison, and diagnostics view
│   ├── prediction.py               # Live what-if inference view
│   ├── history.py                  # Prediction history log and export view
│   └── about.py                    # Project background & academic specifications view
│
└── assets/
    └── style.css                   # Custom CSS design system for clean styling
```

---

## 9. Installation

### Step 1: Clone or Navigate to the Project Directory
```bash
cd "e:\Bhanu Gupta\ml_project"
```

### Step 2: Create a Virtual Environment (Recommended)
**Windows (PowerShell / Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 10. How to Run

Launch the Streamlit web application with:

```bash
streamlit run app.py
```

Streamlit will start a local web server (usually at `http://localhost:8501`) and automatically open it in your default web browser.

To run the standalone backend test script:
```bash
python test_pipeline.py
```

---

## 11. How the ML Model Works

```
[User Input Features] 
        │
        ▼
[Scikit-Learn Pipeline]
  ├── Categorical Features (Day_of_Week, Is_Weekend) ──> OneHotEncoder (Sparse=False)
  └── Numerical Features (Temp, Humidity, People, etc.) ──> StandardScaler (Z-Score Normalization)
        │
        ▼
[Best Estimator: Random Forest Regressor]
  ├── Ensemble of 100 Decision Trees
  └── Final Output = Mean prediction across all trees
        │
        ▼
[Post-Processing & Business Logic]
  ├── Predicted Value in kWh (rounded to 2 decimal places)
  ├── Estimated domestic cost at ₹7.50 / kWh
  ├── Anomaly Check: If predicted kWh > 45 kWh ──> Trigger High Consumption Warning
  └── SQLite DB Persistence: Logged to predictions.db
```

---

## 12. Evaluation Metrics

Because this is a continuous regression problem, the models are evaluated on an unseen 20% test set using:

| Metric | Formula | Physical Interpretation |
| :--- | :--- | :--- |
| **MAE** | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$ | Mean Absolute Error in $kWh$. Lower is better. |
| **MSE** | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | Mean Squared Error. Penalizes large deviations heavily. |
| **RMSE** | $\sqrt{\text{MSE}}$ | Root Mean Squared Error in $kWh$. Lower is better. |
| **$R^2$ Score** | $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$ | Proportion of variance explained by model (1.0 = perfect fit). |

### Typical Benchmark Results on Unseen Test Set:

| Model | MAE (kWh) | MSE | RMSE (kWh) | $R^2$ Score | Selection |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 2.84 | 11.78 | 3.43 | 0.9226 | Baseline |
| **Decision Tree Regressor** | 2.26 | 7.95 | 2.82 | 0.9477 | Non-linear Tree |
| **Random Forest Regressor** | **1.44** | **3.44** | **1.85** | **0.9774** | 🏆 **Best Generalization** |

*Academic Note:* We explicitly explain in the UI and documentation that $R^2$ is the **Coefficient of Determination** and should not be casually called "Accuracy", as accuracy is reserved for classification percentage correctness.

---

## 13. Screenshots & UI Walkthrough

### 1. Home / Dashboard
* Clean summary KPI cards showing total records, baseline usage, summer peak load, idle load, and best model $R^2$.
* Dual interactive charts showing 90-day trends and temperature scatter.
* Quick-access recent prediction audit table.

### 2. Dataset Explorer & CSV Upload
* Tabular preview with adjustable sample rows.
* Summary statistics (count, mean, std, quartiles).
* Clear academic explanation of feature impacts.
* CSV uploader validating schema requirements with sample CSV download button.

### 3. Data Visualization
* Interactive Plotly charts with zoom, hover tooltips, and date range sliders.
* Thermal vs. consumption hockey-stick response curve.
* Weekday average bar charts and Pearson correlation matrix.

### 4. Model Training & Comparison
* Split slider and one-click training button.
* Benchmark comparison table with conditional formatting.
* Grouped bar chart comparing MAE, RMSE, and $R^2$.
* Actual vs. Predicted scatter plot with $45^\circ$ dashed identity line.
* Feature importance rankings showing temperature, appliance hours, and occupancy dominance.

### 5. Predict Consumption
* Clean two-column scenario form for environmental and household parameters.
* Large prediction display in kWh with domestic cost estimation.
* Automated **"High Consumption Warning"** banner for values $> 45\text{ kWh}$.
* Dropdown to toggle between Linear Regression, Decision Tree, and Random Forest predictions.

### 6. Prediction History
* Real-time query of SQLite records.
* Model-based filtering, CSV export, and clear database button.

### 7. About Project
* Complete theoretical overview, algorithm explanations, and architecture diagrams.

---

## 14. Future Scope
* **Hardware Smart Meter Telemetry:** Ingest live continuous readings via MQTT protocols or ESP32 energy monitors.
* **Weather API Integration:** Connect live 7-day weather forecasts via OpenWeatherMap API for automated future predictions.
* **Time-Series Forecasting:** Implement statistical models (ARIMA / SARIMAX) or Prophet to contrast with supervised regression.
* **Energy-Saving Recommendation Engine:** Provide actionable tips (e.g., shifting washing machine runs to off-peak hours).
* **Mobile / Cloud Deployment:** Containerize with Docker and deploy to Streamlit Community Cloud or AWS EC2.

---

## 15. Limitations
* **Synthetic Demonstrative Dataset:** The provided dataset is modeled after physical heuristics and daily residential patterns rather than multi-submetered commercial sensors.
* **Point Estimation:** Predictions provide point values without confidence intervals.
* **Aggregate Appliances:** Appliance usage is tracked in cumulative hours rather than sub-metered per device (refrigerator, HVAC, induction stove).

---

## 16. Conclusion
The **Electricity Consumption Predictor** successfully illustrates how supervised machine learning regression algorithms can be applied to real-world energy analytics. By comparing Linear Regression, Decision Tree Regressor, and Random Forest Regressor within an end-to-end Scikit-Learn pipeline and Streamlit interface, this project serves as a comprehensive, viva-ready, and practical capstone for a 3rd-year B.Tech engineering curriculum.
