import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        "Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized": "schizophrenia",
        "Depressive disorders (share of population) - Sex: Both - Age: Age-standardized": "depression",
        "Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized": "anxiety",
        "Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized": "bipolar",
        "Eating disorders (share of population) - Sex: Both - Age: Age-standardized": "eating"
    })
    df["high_depression"] = (df["depression"] > df["depression"].median()).astype(int)
    return df
