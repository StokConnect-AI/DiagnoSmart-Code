from sklearn.linear_model import LogisticRegression

def train_logistic(X_train_scaled, y_train):
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    return log_reg
