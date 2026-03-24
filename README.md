# Superstore Customer Response Prediction - Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

 ### Project Overview
This project implements an end-to-end machine learning pipeline to predict customer responses to marketing campaigns using the Superstore dataset. The pipeline includes data preprocessing, feature engineering, model building, hyperparameter tuning, and comprehensive evaluation.

### Objective
Analyze customer demographics, purchase history, and engagement behavior to:

Build predictive models for marketing campaign response
. Identify key features influencing customer decisions
. Optimize targeted marketing strategies
. Improve customer engagement effectiveness

### Dataset Contain
File: superstore_data.csv

Size: 2,240 records × 22 features

Key Features:
Demographics: Age (Year_Birth), Education, Marital Status
Purchase History: Wine, Fruits, Meat, Fish, Sweet Products, Gold Products
Engagement: Web Purchases, Catalog Purchases, Store Purchases, Website Visits
Behavioral: Recency, Number of Deals Purchases, Complaints
Target Variable: Response (Binary: 0/1)

### Libraries:
pandas - Data manipulation
scikit-learn - ML algorithms & preprocessing
xgboost - Gradient boosting
matplotlib & seaborn - Visualization
numpy - Numerical operations


### Pipeline Architecture
1. Data Preprocessing
Handle Missing Values
- SimpleImputer with mean strategy for numerical features
- SimpleImputer with most_frequent strategy for categorical features

Feature Engineering
- Convert Dt_Customer to datetime
- Extract Customer_Year and Customer_Month
- Drop original datetime column

Data Splitting
- Train-Test Split: 80-20
- Random State: 42 (reproducibility)

2. Feature Engineering
Numeric Features: Scaled using StandardScaler
Categorical Features: One-Hot Encoded
Column Transformer: Separate pipelines for numeric and categorical data

3. Model Building
Model 1: Random Forest Classifier
GridSearchCV Parameters:
- n_estimators: [50, 100]
- max_depth: [None, 10]
- CV Folds: 3

Model 2: Logistic Regression
Configuration:
- Preprocessor: ColumnTransformer
- Max Iterations: 1000
- Regularization: L2 (default)

Model 3: XGBoost Classifier
GridSearchCV Parameters:
- n_estimators: [100, 200]
- max_depth: [3, 5]
- learning_rate: [0.01, 0.1]
- eval_metric: logloss
- CV Folds: 3

4. Model Evaluation
Metrics Used:
Accuracy: Overall correctness of predictions
Precision: True Positives / (True Positives + False Positives)
Recall: True Positives / (True Positives + False Negatives)
F1 Score: Harmonic mean of Precision and Recall


Model Performance Comparison
Model	Accuracy	Precision	Recall	F1 Score
Random Forest	85.71%	83.56%	85.71%	83.66%
Logistic Regression	85.49%	83.21%	85.49%	83.33%
XGBoost (Tuned)	83.48%	81.69%	83.48%	82.36%   

Best Model: Random Forest
Best Parameters: n_estimators=100, max_depth=10
Accuracy: 85.71%
Reasoning: Best balance of precision and recall; handles non-linear relationships well


## Installation
pip install pandas scikit-learn xgboost matplotlib seaborn numpy
Running the Pipeline
-Load and preprocess data
import pandas as pd
df = pd.read_csv('superstore_data.csv')

-Run feature engineering and model training
-(See notebook: final_ml_pipeline.ipynb)

-Make predictions
predictions = best_model.predict(X_test)

### File Structure
├── final_ml_pipeline.ipynb       # Main notebook with full pipeline
├── superstore_data.csv           # Input dataset
└── README.md                     # This file

### Key Insights
Feature Importance: Purchase history and customer demographics are strong predictors
Class Imbalance: Target variable shows imbalance (85% Class 0 vs 15% Class 1)
Model Performance: Random Forest outperforms other models
Business Impact: 85.71% accuracy enables targeted marketing to high-responding customers

### References
Scikit-learn Documentation: https://scikit-learn.org/
XGBoost Documentation: https://xgboost.readthedocs.io/
Pandas Documentation: https://pandas.pydata.org/
