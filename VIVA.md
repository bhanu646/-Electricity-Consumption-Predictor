# 🎓 Viva Voce & Interview Preparation Guide
## Electricity Consumption Predictor

This document contains clear, simple, and exam-ready answers to the 25 most frequently asked viva and interview questions for this B.Tech Machine Learning project.

---

### Q1. What is Machine Learning?
**Answer:**
Machine Learning (ML) is a subfield of Artificial Intelligence (AI) where algorithms learn mathematical patterns and relationships directly from historical data to make predictions or decisions on unseen future data, without being explicitly programmed with hardcoded rules.

---

### Q2. What is Supervised Learning?
**Answer:**
Supervised Learning is a category of Machine Learning where the training dataset contains both:
1. **Input features ($X$):** The explanatory variables (e.g., Temperature, Number of People, Appliance Hours).
2. **Ground-truth labels / Target ($Y$):** The expected output (e.g., Electricity Consumption in kWh).

The algorithm's objective is to learn a mapping function $f(X) \approx Y$ such that it can predict $Y$ accurately when presented with new inputs $X$.

---

### Q3. What is Regression?
**Answer:**
Regression is a supervised learning task where the target output variable is **continuous and numerical** (e.g., predicting temperature, stock prices, or electricity consumption in kWh), as opposed to classification where the target is a discrete categorical label (e.g., Spam vs. Not Spam).

---

### Q4. Why is this project a regression problem?
**Answer:**
Because our objective is to predict **daily electricity consumption**, which is a continuous physical quantity measured in kilowatt-hours (kWh) (e.g., $18.42\text{ kWh}$, $32.80\text{ kWh}$). Since the output can take any real number value rather than a finite set of categories, it is formulated as a regression problem.

---

### Q5. What is Linear Regression?
**Answer:**
Linear Regression is a fundamental parametric regression algorithm that models the relationship between dependent variable $Y$ and one or more independent variables $X$ as a linear equation:
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \epsilon$$
It estimates the coefficients $\beta$ using **Ordinary Least Squares (OLS)** by minimizing the sum of squared differences between actual and predicted values.

---

### Q6. What is Decision Tree Regression?
**Answer:**
Decision Tree Regression is a non-parametric algorithm that breaks down the dataset into smaller subsets by learning binary decision rules (if-else conditions) based on feature values. In regression, splits are chosen to minimize the variance (Mean Squared Error) within the resulting leaf nodes. A query point's prediction is the average target value of training samples inside that leaf.

---

### Q7. What is Random Forest?
**Answer:**
Random Forest is an ensemble learning method based on **Bootstrap Aggregating (Bagging)**. It builds a forest of multiple decision trees (e.g., 100 trees), where each tree is trained on a random bootstrap sample with a random subset of features at each split. The final prediction is the average of predictions across all individual trees, which drastically reduces model variance and prevents overfitting.

---

### Q8. Why did you use these algorithms?
**Answer:**
1. **Linear Regression:** Acts as a transparent baseline model to establish benchmark performance and interpret basic linear effects.
2. **Decision Tree Regressor:** Captures non-linear thresholds (e.g., air conditioners only kick in above 24°C).
3. **Random Forest Regressor:** Combines the strengths of decision trees while reducing variance and overfitting, yielding the highest accuracy and stability for tabular data.
Comparing all three allows us to demonstrate how model complexity affects bias and variance.

---

### Q9. What is Training Data?
**Answer:**
Training data is the portion of the dataset (typically 80%) used by the machine learning algorithm to learn parameters, patterns, and weights during the fitting process (`model.fit(X_train, y_train)`).

---

### Q10. What is Testing Data?
**Answer:**
Testing data is the held-out portion of the dataset (typically 20%) that the model has **never seen** during training. It is used strictly for unbiased evaluation (`model.predict(X_test)`) to assess how well the model generalizes to new, real-world data.

---

### Q11. What is Feature Engineering?
**Answer:**
Feature Engineering is the process of using domain knowledge to extract new informative variables or transform existing raw variables to improve model learning. In this project, we extracted `Month`, `Day`, `Day_of_Week`, and `Is_Weekend` from the raw `Date` string to capture seasonal climate cycles and weekend occupancy surges.

---

### Q12. What is Data Preprocessing?
**Answer:**
Data preprocessing is the phase where raw data is cleaned, validated, and transformed into a format suitable for ML algorithms. It involves:
1. Handling missing values (median imputation for numerical features).
2. Removing duplicate records.
3. Encoding categorical features (One-Hot Encoding for `Day_of_Week`).
4. Feature scaling (StandardScaler to normalize numeric scales for Linear Regression).
5. Preventing data leakage by encapsulating steps inside a Scikit-Learn `Pipeline`.

---

### Q13. What is MAE (Mean Absolute Error)?
**Answer:**
MAE is the average of the absolute differences between actual values and predicted values:
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$
It represents the average expected error magnitude in the same unit as the target (kWh). An MAE of $1.44\text{ kWh}$ means predictions deviate by approximately $1.44\text{ kWh}$ on average.

---

### Q14. What is MSE (Mean Squared Error)?
**Answer:**
MSE is the average of the squared differences between actual and predicted values:
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$
Because errors are squared, MSE penalizes large errors much more severely than small errors.

---

### Q15. What is RMSE (Root Mean Squared Error)?
**Answer:**
RMSE is the square root of MSE:
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
RMSE penalizes large outlier errors like MSE, but by taking the square root, it converts the error metric back into the original unit of measurement (kWh).

---

### Q16. What is R² Score (Coefficient of Determination)?
**Answer:**
$R^2$ measures the proportion of variance in the dependent variable that is predictable from the independent features:
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
* An $R^2$ of $1.0$ represents a perfect prediction.
* An $R^2$ of $0.0$ means the model performs no better than simply predicting the mean.
* Note: In regression, we do not call $R^2$ "accuracy" because accuracy is reserved for classification percentage correctness.

---

### Q17. What is Overfitting?
**Answer:**
Overfitting occurs when a model learns the training data and random noise too closely, performing exceptionally well on training data ($R^2 \approx 1.0$) but failing to generalize on unseen test data. A deep decision tree with no depth limits is an example of an overfitted model.

---

### Q18. What is Underfitting?
**Answer:**
Underfitting occurs when a model is too simple to capture the underlying structure and patterns of the data, resulting in poor performance on both training and test datasets. An example is fitting a straight line to highly non-linear data.

---

### Q19. Why do we split the dataset into Train and Test sets?
**Answer:**
We split the dataset (e.g., 80% train, 20% test) to evaluate model performance on **unseen data**. If we evaluated the model on the data it was trained on, we would get an artificially optimistic assessment that masks overfitting. The test set simulates real-world production deployment.

---

### Q20. Which model performed best in your project and why?
**Answer:**
**Random Forest Regressor** performed best, achieving an $R^2$ score of approximately **0.977** (compared to ~0.923 for Linear Regression and ~0.948 for a single Decision Tree) and the lowest RMSE (~$1.85\text{ kWh}$).
It won because:
1. It handles complex non-linear relationships (such as cooling loads increasing non-linearly with temperature).
2. Its bagging ensemble mechanism averages across 100 decorrelated trees, mitigating individual tree variance and noise.

---

### Q21. What is the Target Variable in this project?
**Answer:**
The target variable is `Electricity_Consumption` (measured continuously in kilowatt-hours, kWh).

---

### Q22. What are the Input Features?
**Answer:**
1. `Temperature` (°C)
2. `Humidity` (%)
3. `Number_of_People` (Occupancy count)
4. `Previous_Day_Consumption` (Yesterday's kWh)
5. `Appliance_Usage` (Major appliance operating hours)
6. `Day_of_Week` (Monday–Sunday)
7. `Is_Weekend` (Binary 0 or 1)
8. `Month` (Extracted from Date)

---

### Q23. How does the prediction pipeline work in production?
**Answer:**
When a user submits input values on the Streamlit UI:
1. The inputs are structured into a 1-row Pandas DataFrame.
2. The DataFrame passes through the Scikit-Learn `ColumnTransformer` (standardizing numerical features and one-hot encoding categorical ones).
3. The trained estimator generates a continuous output in kWh.
4. If the predicted value exceeds the 90th percentile threshold (~45 kWh), an automated "High Consumption Warning" is triggered.
5. The transaction is logged with a timestamp into the SQLite database (`data/predictions.db`).

---

### Q24. What are the limitations of the project?
**Answer:**
1. **Synthetic Data:** The dataset is generated through physical heuristics rather than hardware smart meters.
2. **Point Estimates:** The model provides point predictions without Bayesian prediction intervals or uncertainty bounds.
3. **Appliance Breakdown:** Appliances are aggregated into cumulative hours rather than sub-metering individual devices (e.g., AC vs. Refrigerator vs. EV Charger).

---

### Q25. What is the future scope of this project?
**Answer:**
1. Integrating IoT smart energy meters via MQTT or REST APIs for real-time live telemetry.
2. Integrating real-time weather forecasting APIs (e.g., OpenWeatherMap) to forecast 7-day ahead electricity demands.
3. Implementing time-series models (ARIMA/SARIMAX/Prophet) to compare against regression models.
4. Adding an automated recommendation engine suggesting thermostat setbacks and load shifting during peak tariff hours to save electricity costs.
