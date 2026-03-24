# Superstore Customer Response Prediction - Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📋 Project Overview

This project implements an **end-to-end machine learning pipeline** to predict customer responses to marketing campaigns using the Superstore dataset. The pipeline analyzes customer demographics, purchase history, and engagement behavior to identify high-potential customers for targeted marketing.

### Key Achievements
- ✅ **85.71% Accuracy** with Random Forest Classifier
- ✅ **Precision: 83.57%** for reliable positive predictions
- ✅ **Production-ready pipeline** with preprocessing and feature engineering
- ✅ **Hyperparameter tuning** using GridSearchCV

---

## 🎯 Problem Statement

Marketing teams spend significant resources on campaigns with low response rates. This project solves that by:
- Predicting which customers will respond to marketing campaigns
- Identifying key customer segments
- Optimizing marketing budget allocation
- Improving customer engagement strategies

---

## 📊 Dataset

**Source**: Superstore Customer Dataset
**Size**: 2,240 customer records with 22 features
**Target Variable**: `Response` (Binary: 0 = No, 1 = Yes)

### Features Included:
- **Demographics**: Age (Year_Birth), Education, Marital Status
- **Financial**: Income, Purchase amounts by category
- **Behavior**: Number of purchases (Web, Catalog, Store), Website visits
- **Engagement**: Days since last purchase (Recency), Complaints
- **Product Categories**: Wines, Fruits, Meat, Fish, Sweets, Gold

---

##Architecture

### Pipeline Components:

1. **Data Preprocessing**
   - Missing value imputation (mean for numerical, most_frequent for categorical)
   - Outlier detection and handling
   - Categorical encoding (OneHotEncoder)
   - Feature scaling (StandardScaler)

2. **Feature Engineering**
   - Date feature extraction (Year, Month from registration date)
   - Customer segmentation features
   - RFM (Recency, Frequency, Monetary) analysis

3. **Model Training**
   - **Random Forest Classifier** (Primary Model)
   - **Logistic Regression** (Baseline Model)
   - **XGBoost** (Advanced Model - Optional)
   - Hyperparameter optimization with GridSearchCV

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - ROC-AUC Curve
   - Feature Importance Analysis

---

## 📈 Model Performance

### Random Forest Results (Best Model)
