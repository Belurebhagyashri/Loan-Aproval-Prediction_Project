# Loan-Aproval-Prediction_Project
📘 Loan Approval Prediction
📌 Project Overview

This project predicts whether a loan application will be approved or rejected using machine learning.
We perform Exploratory Data Analysis (EDA), build ML models, and evaluate their performance.

⚙️ Tech Stack Used

Python 🐍

Pandas, NumPy → Data handling

Matplotlib, Seaborn → Visualization

Scikit-learn → ML models & evaluation

SHAP → Model explainability

📂 Project Workflow
🔹 Step 1: Import Libraries

All required libraries for data analysis, visualization, and ML are imported.

🔹 Step 2: Load Data

We load the dataset (loan.csv or given file) and check:

Shape of data

Missing values

Basic statistics

🔹 Step 3: Exploratory Data Analysis (EDA)

Visualize loan approval trends.

Check categorical features (Gender, Education, Credit History).

Understand correlations between variables.

🔹 Step 4: Data Preprocessing

Handle missing values.

Encode categorical variables.

Scale numerical features.

Split into Train (80%) and Test (20%).

🔹 Step 5: Model Building

We train two models:

Logistic Regression (baseline model)

Random Forest Classifier (better accuracy).

🔹 Step 6: Model Evaluation

Confusion Matrix → Correct vs wrong predictions.

Classification Report → Precision, Recall, F1-score.

ROC AUC Score → Model performance.

SHAP Plots → Feature importance (e.g., Credit History, Income).

📊 Results

Logistic Regression → Good baseline.

Random Forest → Higher accuracy.

Most important features: Credit History, Applicant Income, Loan Amount.

🚀 How to Run

Clone the repo / download the notebook.

Install requirements:

pip install pandas numpy matplotlib seaborn scikit-learn shap


Open the Jupyter Notebook:

jupyter notebook Loan_Approval_Prediction.ipynb


Run each cell step by step.

✅ Key Learning

How to clean and preprocess real-world loan data.

Train ML models for classification.

Evaluate using multiple metrics.

Explain predictions with SHAP values.
