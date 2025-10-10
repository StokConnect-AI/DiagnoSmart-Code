import pandas as pd

def custom_prediction(log_reg, rf, scaler):
    print("\n--- Custom Prediction ---")
    try:
        # Ask for patient's name
        patient_name = input("Enter the patient's name: ").strip().title()

        # Ask for disorder prevalence inputs
        s = float(input("Enter schizophrenia prevalence: "))
        a = float(input("Enter anxiety prevalence: "))
        b = float(input("Enter bipolar prevalence: "))
        e = float(input("Enter eating disorder prevalence: "))

        # Warn if values are unusually high
        for val, name in zip([s, a, b, e], ["schizophrenia", "anxiety", "bipolar", "eating"]):
            if val > 10:
                print(f"Warning: {name} value {val} seems outside the normal range (0–5%).")

        # Build input DataFrame
        features = pd.DataFrame([{
            "schizophrenia": s,
            "anxiety": a,
            "bipolar": b,
            "eating": e
        }])

        # Predict
        features_scaled = scaler.transform(features)
        pred_lr = log_reg.predict(features_scaled)[0]
        pred_rf = rf.predict(features)[0]

        # Interpret results
        def interpret(pred):
            return "more likely to be suffering from mental illness" if pred == 1 else "less likely to be suffering from mental illness"

        print(f"\nPrediction for {patient_name}:")
        print(f"• Logistic Regression suggests {patient_name} is {interpret(pred_lr)}.")
        print(f"• Random Forest suggests {patient_name} is {interpret(pred_rf)}.")

        #Log the patient data
        log_new_patient(patient_name, s, a, b, e, pred_lr, pred_rf)

    except Exception as ex:
        print("Prediction skipped. Error:", ex)

def log_new_patient(patient_name, s, a, b, e, pred_lr, pred_rf):
    row = {
        "patient_name": patient_name,
        "schizophrenia": s,
        "anxiety": a,
        "bipolar": b,
        "eating": e,
        "logistic_prediction": pred_lr,
        "random_forest_prediction": pred_rf
    }

    try:
        df = pd.read_csv("NewPatients.csv")
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame([row])

    df.to_csv("NewPatients.csv", index=False)
    print(f"New patient data for {patient_name} saved to NewPatients.csv")
