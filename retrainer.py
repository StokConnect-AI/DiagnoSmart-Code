#This script is a well-structured pipeline for retraining two machine learning models, 
#Logistic Regression and Random Forest Classifier to predict whether a patient has a high likelihood of depression based on other mental health indicators.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def retrain_models(original_path="1- mental-illnesses-prevalence.csv", new_path="NewPatients.csv"):
    # Load and clean original dataset
    df = pd.read_csv(original_path)
    df = df.rename(columns={
        "Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized": "schizophrenia",
        "Depressive disorders (share of population) - Sex: Both - Age: Age-standardized": "depression",
        "Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized": "anxiety",
        "Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized": "bipolar",
        "Eating disorders (share of population) - Sex: Both - Age: Age-standardized": "eating"
    })
    df["high_depression"] = (df["depression"] > df["depression"].median()).astype(int)
    df = df[["schizophrenia", "anxiety", "bipolar", "eating", "high_depression"]]

    # Load and validate new patient data
    try:
        new_df = pd.read_csv(new_path)
        new_df["high_depression"] = new_df["logistic_prediction"].astype(int)
        new_df = new_df[["schizophrenia", "anxiety", "bipolar", "eating", "high_depression"]]

        if not new_df.empty and not new_df.isna().all().all():
            df = pd.concat([df, new_df], ignore_index=True)
            print("✅ Merged original and new patient data")
        else:
            print("⚠️ New patient data is empty or invalid. Skipping merge.")
    except FileNotFoundError:
        print("⚠️ No new patient data found. Using original dataset only.")
    except pd.errors.EmptyDataError:
        print("⚠️ Patient file is empty. Using original dataset only.")

    # Prepare features and labels
    X = df[["schizophrenia", "anxiety", "bipolar", "eating"]]
    y = df["high_depression"].astype(int)

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    print("✅ Models retrained with updated data")
    return log_reg, rf, scaler

