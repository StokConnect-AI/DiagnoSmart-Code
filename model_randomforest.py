#This Python script defines a function that trains a Random Forest classifier using 200 decision trees on the provided training data. 
#The model is initialized with a fixed random seed for reproducibility, trained on the input features and labels, and then returned for future predictions or evaluation.

from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    return rf


