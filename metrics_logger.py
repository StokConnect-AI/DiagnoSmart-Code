import pandas as pd
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def log_metrics(model_name, y_true, y_pred):
    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

    hist_file = "History.csv"
    try:
        hist_df = pd.read_csv(hist_file)
        hist_df = pd.concat([hist_df, pd.DataFrame([row])], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        hist_df = pd.DataFrame([row])

    hist_df.to_csv(hist_file, index=False)
    print(f"✅ Metrics for {model_name} saved to {hist_file}")
