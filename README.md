# Loan Default Prediction using Gradient Boosting Decision Trees

## 🔹 Introduction

This project builds a binary classifier to predict loan defaults (whether a customer will default on a two-wheeler loan) using a real-world TVS Credit Services dataset. The project applies multiple feature selection strategies, compares classical models (Decision Tree, Random Forest, Logistic Regression, Naive Bayes) against gradient boosting approaches (CatBoost), and identifies the best model by AUC score.

**Main notebook:** `loan2.ipynb`
**Requirements:** `requirements.txt`

## 🔹 Objectives

- Predict loan default (binary: default vs. non-default) for two-wheeler loan customers.
- Handle extensive missing data and engineer useful features for classification.
- Compare multiple models and feature selection strategies across 3 experimental setups.
- Maximize AUC-ROC as the primary evaluation metric.

## 🔹 Dataset & Features

- **Source:** TVS Credit Services — TVS Loan Default dataset (local CSV)
- **Records:** 119,528 rows × 32 columns (V1–V32)
- **Target:** V32 — Loan default label (0 = No default, 1 = Default)
- **Key Features (selected):**
  - V2: Bounced on first EMI, V3: Bounces in last 12 months
  - V6: EMI amount, V7: Loan Amount, V8: Tenure
  - V13: Gender, V14: Employment type, V15: Resident type
  - V18: Number of loans, V28–V31: Advance EMI payments, Tier classification
- **Challenge:** Heavy missing data — many columns had up to 34,480 missing values; dropped entirely after assessment.

## 🔹 EDA Highlights

  - Dataset has many "?" values (treated as NaN) and a highly imbalanced target (default ≪ non-default).
  - Many columns dropped due to high missingness.
  - ANOVA (f_classif) used for feature selection — outperformed outlier- and correlation-based methods for this imbalanced dataset.
  - Three feature strategies tested: outlier-based (S1), correlation-based (S2), and ANOVA-based (S3).

## 🔹 Modeling

  - **Baseline models (per strategy):** Decision Tree, Random Forest, Logistic Regression, Naive Bayes
  - **Best boosting model:** CatBoost (gradient boosting on categorical/numerical data)
  - **Scaling:** StandardScaler applied for Logistic Regression and Naive Bayes
  - **Evaluation:** AUC-ROC, Precision, Recall, F1-score

## 🔹 Results

  | Model               | Strategy | AUC    | F1     | Precision | Recall |
  |---------------------|----------|--------|--------|-----------|--------|
  | Decision Tree       | S1       | 0.8083 | 0.1374 | 0.0763    | 0.6883 |
  | Decision Tree       | S3       | 0.8153 | 0.1456 | 0.0813    | 0.6941 |
  | Random Forest       | S1       | 0.8039 | 0.1617 | 0.0927    | 0.6348 |
  | **CatBoost**        | —        | **0.8418** | — | —      | —      |

  ✅ **Best model: CatBoost with AUC = 0.8418** — highest discriminative power among all tested models.

## 🔹 Conclusion

1. Heavy missing data required significant column dropping before modeling.
2. ANOVA-based feature selection outperformed outlier and correlation methods for this imbalanced dataset.
3. Gradient Boosting (CatBoost) is best suited for this type of tabular, imbalanced data.
4. CatBoost achieved AUC = 0.8418, the best across all strategies and models.
5. Future work: SMOTE for oversampling, additional feature engineering, hyperparameter tuning.

## 🔹 Executive Summary
This project predicts two-wheeler loan defaults using the TVS Credit dataset (119,528 records, 32 features). After extensive data cleaning (missing values, "?" handling) and three different feature selection strategies (outlier, correlation, ANOVA), four classifiers were compared alongside gradient boosting. CatBoost emerged as the best model with AUC = 0.8418, outperforming tree-based and linear models. The ANOVA-based feature selection strategy (S3) produced the most reliable features for separating defaulters from non-defaulters in this cost-sensitive setting.
