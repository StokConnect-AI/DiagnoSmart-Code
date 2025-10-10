import pandas as pd
from config import FEATURE_COLUMNS, NEW_PATIENTS_PATH
from utils import warn_if_out_of_range
from speech_input import get_input, choose_input_mode

def log_new_patient(patient_id, patient_name, s, a, b, e, pred_lr, pred_rf):
    row = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "schizophrenia": s,
        "anxiety": a,
        "bipolar": b,
        "eating": e,
        "logistic_prediction": pred_lr,
        "random_forest_prediction": pred_rf
    }

    try:
        df = pd.read_csv(NEW_PATIENTS_PATH)
        df["patient_id"] = df["patient_id"].astype(str).str.zfill(13)

        if patient_id in df["patient_id"].values:
            for col, val in row.items():
                df.loc[df["patient_id"] == patient_id, col] = val
            print(f"🔄 Updated existing patient record for ID {patient_id}")
        else:
            new_row_df = pd.DataFrame([row])
            if df.empty or df.isna().all().all():
                df = new_row_df
            else:
                df = pd.concat([df, new_row_df], ignore_index=True)
            print(f"🆕 Added new patient record for ID {patient_id}")

    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame([row])
        print(f"🆕 Created new patient file with first entry for ID {patient_id}")

    df.to_csv(NEW_PATIENTS_PATH, index=False)
    print(f"✅ Patient data for {patient_name} saved to Patient File")

def custom_prediction(log_reg, rf, scaler):
    print("\n--- Custom Prediction ---")
    choose_input_mode()

    try:
        # Ask for patient's name
        patient_name = get_input("Enter the patient's name:").title()

        # Ask for patient's 13-digit numeric South African ID
        while True:
            patient_id = get_input("Enter the patient's South African ID:", str).strip()
            if patient_id.isdigit() and len(patient_id) == 13:
                break
            print("⚠️ Invalid ID. Please enter exactly 13 digits (numbers only).")

        # Ask for disorder prevalence inputs
        s = get_input("Enter schizophrenia prevalence:", float)
        a = get_input("Enter anxiety prevalence:", float)
        b = get_input("Enter bipolar prevalence:", float)
        e = get_input("Enter eating disorder prevalence:", float)

        # Warn if values are unusually high
        for val, name in zip([s, a, b, e], FEATURE_COLUMNS):
            warn_if_out_of_range(val, name)

        # Build input DataFrame
        features = pd.DataFrame([dict(zip(FEATURE_COLUMNS, [s, a, b, e]))])
        features_scaled = scaler.transform(features)

        # Predict probabilities
        prob_lr = log_reg.predict_proba(features_scaled)[0][1]
        prob_rf = rf.predict_proba(features)[0][1]

        # Combine probabilities
        combined_prob = (prob_lr + prob_rf) / 2

        #Override if all inputs are zero
        if all(v == 0 for v in [s, a, b, e]):
            final_pred = 0
            combined_prob = 0.0
        else:
            final_pred = int(combined_prob >= 0.5)

        # Format output
        result = "more likely" if final_pred == 1 else "less likely"
        percent = int(round(combined_prob * 100))

        # Unified output
        print(f"\nPrediction for {patient_name}:")
        print(f"{patient_name} is {result} to be suffering from mental illness.")
        print(f"Estimated likelihood: {percent}%")
        print("⚠️ Disclaimer: This prediction is a supportive tool. Final diagnosis and treatment decisions must be made by a qualified healthcare professional.")


        # Log the patient data
        pred_lr = int(prob_lr >= 0.5)
        pred_rf = int(prob_rf >= 0.5)
        log_new_patient(patient_id, patient_name, s, a, b, e, pred_lr, pred_rf)

    except Exception as ex:
        print("Prediction skipped. Error:", ex)
        
