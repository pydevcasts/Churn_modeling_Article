# Customer Churn Prediction in Banking

## Advanced Feature Engineering & Ensemble Learning for Imbalanced Banking Data

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-red.svg)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Overview

This project presents a robust machine learning solution for **customer churn prediction** in the banking sector. The model addresses the challenge of highly imbalanced datasets (10:1 ratio) through advanced feature engineering and ensemble learning techniques.

### Key Highlights
- **Accuracy**: 92%
- **AUC Score**: 0.96
- **Precision**: 0.96
- **Recall**: 0.87
- **Outperformed previous best model** (91% accuracy)

---

## 🎯 Problem Statement

Customer churn is a critical challenge in the banking industry. Identifying customers likely to leave enables proactive retention strategies. The main challenges addressed:
- **Highly imbalanced dataset** (90% non-churn vs 10% churn)
- **Complex feature interactions** affecting customer behavior
- **Need for interpretable** yet powerful predictions

---

## 📊 Dataset

The dataset contains banking customer information with the following features:

| Feature | Description |
|---------|-------------|
| CreditScore | Customer's credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Male/Female |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Credit card ownership (0/1) |
| IsActiveMember | Active member status (0/1) |
| EstimatedSalary | Estimated annual salary |
| Exited | **Target**: Churn status (1 = exited) |

---

## 🛠️ Methodology

### 1. Data Preprocessing

**Outlier Removal:**
```python
- CreditScore ≤ 359
- Age ≥ 71 years
- NumOfProducts ≥ 4
