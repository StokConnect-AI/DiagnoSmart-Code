from retrainer import retrain_models
from metrics_logger import log_metrics
from visualizer import plot_disorder_trend
from predictor import custom_prediction
from data_loader import load_and_clean_data
from preprocessor import prepare_data

# Step 1: Load original data
df = load_and_clean_data("1- mental-illnesses-prevalence.csv")

# Step 2: Retrain models using original + new patient data
log_reg, rf, scaler = retrain_models()

# Step 3: Evaluate and log metrics using original data
X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, _ = prepare_data(df)
log_metrics("Logistic Regression", y_test, log_reg.predict(X_test_scaled))
log_metrics("Random Forest", y_test, rf.predict(X_test))

# Step 4: Custom prediction with patient name and logging
custom_prediction(log_reg, rf, scaler)

# Step 5: Ask if user wants to see prevalence trends
choice = input("\nWould you like to view prevalence trends by country/entity? (yes/no): ").strip().lower()

if choice in ["yes", "y"]:
    print("\nAvailable countries/entities:")
    print(df["Entity"].unique())
    entity = input("\nEnter the country/entity to visualize prevalence: ").strip()
    disorder = input("Enter the disorder to visualize (schizophrenia, depression, anxiety, bipolar, eating): ").strip()
    plot_disorder_trend(df, entity, disorder)
else:
    print("\nOkay, prediction complete. Exiting now.")
