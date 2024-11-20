# Credit Card Fraud Detection - Machine Learning Models

## Overview
This project implements and compares various Machine Learning models for **Credit Card Fraud Detection**. The goal is to evaluate the performance of each model on both balanced and unbalanced datasets, using key metrics such as Accuracy, Precision, Recall, F1 Score, and ROC AUC. The project also includes data preprocessing, feature scaling, and balancing techniques like SMOTE.

## Features
- **Data Preprocessing:** Standardizes features and balances the training dataset using SMOTE.
- **Model Evaluation:** Compares models based on training and test sets with detailed confusion matrices.
- **Results Analysis:** Saves evaluation metrics in a CSV file for easy analysis.

## Models Used
- Logistic Regression
- Decision Tree Classifier
- Naive Bayes
- XGBoost
- Neural Network (MLP)
- AdaBoost
- Random Forest
- Gradient Boosting

## Prerequisites
- Python 3.7+
- Libraries: 
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `imblearn`
  - `matplotlib`
  - `seaborn`
  - Google Colab (for running the code)

## Dataset
The dataset used is [Kaggle's Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), containing transactions made by European cardholders in September 2013.

## Project Structure
1. **Data Loading:** Reads the dataset from Google Drive.
2. **Data Preprocessing:** Scales numeric features and balances the training dataset using SMOTE.
3. **Model Training:** Trains multiple classifiers on both unbalanced and balanced datasets.
4. **Evaluation:** Computes metrics for each model and saves results to a CSV file.

## How to Run
1. **Set up Google Colab:**
   - Mount your Google Drive using the following command:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Replace the `file_path` variable with the path to the dataset in your Google Drive.
2. **Install Dependencies:**
   Ensure all required Python libraries are installed:
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
