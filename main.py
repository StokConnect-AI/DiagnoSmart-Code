# This file executes the DiagnoSmart workflow:
# - Loads and retrains models using new patient data
# - Logs evaluation metrics to track performance
# - Prompts health workers for patient input via voice or text
# - Predicts the likelihood of mental illness and logs results
# - Optionally visualizes the prevalence of disorders by country
# Final diagnosis decisions should be made only by a qualified healthcare provider.
from retrainer import retrain_models
from metrics_logger import log_metrics
from visualizer import plot_disorder_trend
from predictor import custom_prediction
from data_loader import load_and_clean_data
from preprocessor import prepare_data
from speech_input import get_input, choose_input_mode

# Step 1: Load original data
df = load_and_clean_data("1- mental-illnesses-prevalence.csv")

# Step 2: Retrain models using original + new patient data
log_reg, rf, scaler = retrain_models()

# Print intercept for diagnostic insight
print(f"\nLogistic Regression Intercept: {log_reg.intercept_[0]:.4f}")

# Step 3: Evaluate and log metrics using original data
X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, _ = prepare_data(df)
log_metrics("Logistic Regression", y_test, log_reg.predict(X_test_scaled))
log_metrics("Random Forest", y_test, rf.predict(X_test))

# Step 4: Custom prediction with patient name and logging
custom_prediction(log_reg, rf, scaler)

# Step 5: Ask if user wants to see prevalence trends
print("\n📊 Optional: View prevalence trends by country")
choose_input_mode()

choice = get_input("Would you like to view prevalence trends by country/entity? (yes/no):", str).lower()

if choice in ["yes", "y"]:
    print("\nAvailable countries/entities:")
    print(df["Entity"].unique())
    entity = get_input("\nEnter the country/entity to visualize prevalence:")
    disorder = get_input("Enter the disorder to visualize (schizophrenia, depression, anxiety, bipolar, eating):")
    plot_disorder_trend(df, entity, disorder)
else:
    print("\nOkay, prediction complete. Exiting now.")
