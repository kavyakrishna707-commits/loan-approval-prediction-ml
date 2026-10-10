import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load dataset or create a clean baseline dataset
csv_path = 'loan_data.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    data = {
        'Gender': ['Male', 'Female', 'Male', 'Male', 'Female'] * 10,
        'Married': ['Yes', 'No', 'Yes', 'Yes', 'No'] * 10,
        'Education': ['Graduate', 'Graduate', 'Not Graduate', 'Graduate', 'Graduate'] * 10,
        'Self_Employed': ['No', 'No', 'Yes', 'No', 'No'] * 10,
        'ApplicantIncome': [5000, 3000, 2500, 6000, 4200] * 10,
        'CoapplicantIncome': [1500, 0, 1200, 2000, 0] * 10,
        'LoanAmount': [120, 90, 80, 150, 110] * 10,
        'Loan_Amount_Term': [360, 360, 360, 360, 360] * 10,
        'Credit_History': [1.0, 1.0, 0.0, 1.0, 1.0] * 10,
        'Property_Area': ['Urban', 'Rural', 'Semiurban', 'Urban', 'Urban'] * 10,
        'Loan_Status': ['Y', 'Y', 'N', 'Y', 'Y'] * 10
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

# Clean and convert target column
target_mapping = {'Y': 1, 'N': 0, 'YES': 1, 'NO': 0, '1': 1, '0': 0, 1: 1, 0: 0}
df['Loan_Status'] = df['Loan_Status'].astype(str).str.strip().str.upper().map(target_mapping).fillna(0).astype(int)

X = df.drop(columns=['Loan_ID', 'Loan_Status'], errors='ignore')
y = df['Loan_Status']

numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
categorical_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

# Ensure all columns exist
for col in numeric_features:
    if col not in X.columns:
        X[col] = 0.0
for col in categorical_features:
    if col not in X.columns:
        X[col] = 'Unknown'

X = X[numeric_features + categorical_features]

# Preprocessing Pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X, y)
joblib.dump(pipeline, 'loan_model.pkl')
print("Model pipeline successfully built and saved to loan_model.pkl")