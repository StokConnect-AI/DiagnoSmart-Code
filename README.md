# DiagnoSmart-Code:

**#Features:**
#Prediction for patients using Logistic Regression and Random Forest tailoring.
#Adaptive learning from a new patient file ('NewPatient.csv').
#Trenf visualization by country and disorder.
#Performance metrics logging (accuracy, precision, recall, etc.)

**##File Structure:**
1. main.py: Command Line Interface and Control flow
2. config.py: Contains centralized paths, settings and contraints.
3. data_loader.py: Loads and cleans original dataset along with associated datasets.
4. model_logistic.py: Trains and returns Logistic Regression model.
5. model_randomforest.py: Trains and returns Random Forest Model.
6. retrainer.py: Retrains models with new patient data.
7. predictor.py: Gives prediction and patient logging.
8. metrics_logger.py: It logs model performance metrics.
9. preprocessor.py: Splits, scale and prepares data for training.
10. visualizer.py: It visualizes disorder trends by country.
11. utils.py: Shared helper functions and reusables.
12. speech_input.py: Speech Recognition (Voice input).

**#How To Run The Code:**
run the main.py file
