Credit Card Fraud Detection - Machine Learning Models
Overview
This project implements and compares various Machine Learning models for Credit Card Fraud Detection. The goal is to evaluate the performance of each model on both balanced and unbalanced datasets, using key metrics such as Accuracy, Precision, Recall, F1 Score, and ROC AUC. The project also includes data preprocessing, feature scaling, and balancing techniques like SMOTE.

Features
Data Preprocessing: Standardizes features and balances the training dataset using SMOTE.
Model Evaluation: Compares models based on training and test sets with detailed confusion matrices.
Results Analysis: Saves evaluation metrics in a CSV file for easy analysis.
Models Used
Logistic Regression
Decision Tree Classifier
Naive Bayes
XGBoost
Neural Network (MLP)
AdaBoost
Random Forest
Gradient Boosting
Prerequisites
Python 3.7+
Libraries:
pandas
numpy
scikit-learn
xgboost
imblearn
matplotlib
seaborn
Google Colab (for running the code)
Dataset
The dataset used is Kaggle's Credit Card Fraud Detection Dataset, containing transactions made by European cardholders in September 2013.

Project Structure
Data Loading: Reads the dataset from Google Drive.
Data Preprocessing: Scales numeric features and balances the training dataset using SMOTE.
Model Training: Trains multiple classifiers on both unbalanced and balanced datasets.
Evaluation: Computes metrics for each model and saves results to a CSV file.
How to Run
Set up Google Colab:
Mount your Google Drive using the following command:
python
Copy code
from google.colab import drive
drive.mount('/content/drive')
Replace the file_path variable with the path to the dataset in your Google Drive.
Install Dependencies: Ensure all required Python libraries are installed:
bash
Copy code
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
Run the Code:
Execute the script in a Python environment or Google Colab.
Results will be saved as balancedDataOutcomes.csv in your working directory.
Outputs
Confusion Matrices: Displays confusion matrices for both training and test sets.
Metrics CSV: Saves a detailed summary of evaluation metrics in balancedDataOutcomes.csv.
Key Metrics
True Positives (TP)
True Negatives (TN)
False Positives (FP)
False Negatives (FN)
Accuracy
Precision
Recall
F1 Score
ROC AUC
Sample Results
Classifier Name	Training/Test Set	Balanced/Unbalanced	Accuracy	Precision	Recall	F1 Score	ROC AUC
Logistic Regression	Train	Balanced	0.98	0.96	0.97	0.96	0.99
Random Forest	Test	Unbalanced	0.97	0.94	0.95	0.94	0.98
Contributions
Feel free to fork this repository and contribute improvements!

License
This project is licensed under the MIT License.

