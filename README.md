# Loan Status Prediction Using Machine Learning

A beginner-friendly machine learning project that predicts whether a loan application will be **approved** or **not approved**, using a **Support Vector Machine (SVM)** classifier in scikit-learn.

## Dataset
The notebook expects a file named `loan.csv` (the Loan Prediction dataset). It is not included in this repo, so download it and place it in the project folder.

Features used: Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
Target: `Loan_Status` (1 = approved, 0 = not approved)

## What the notebook does
1. Loads the data and checks its shape, statistics, and missing values
2. Drops rows with missing values
3. Encodes the labels (`Y`/`N` to 1/0) and text columns to numbers
4. Visualizes education and marital status against loan status
5. Splits the data into train/test sets (80/20, stratified)
6. Standardizes features, fitting the scaler on the training set only to avoid data leakage
7. Trains a linear SVM classifier
8. Evaluates it with accuracy and a confusion matrix
9. Builds a simple predictive system for a single applicant

## How to run
```bash
git clone https://github.com/<your-username>/loan-status-prediction.git
cd loan-status-prediction
pip install -r requirements.txt
jupyter notebook Loan_Status_Prediction.ipynb
```

## Tech stack
Python, NumPy, pandas, scikit-learn, seaborn, matplotlib

## Disclaimer
This is a learning project and must not be used to make real lending decisions.
