# ClassifierAlgorithm/custom_logistic_classifier.py

import os
import numpy as np
import pandas as pd
import joblib

# ------------------------------
# SCRATCH LOGISTIC REGRESSION
# ------------------------------
class ScratchLogisticRegression:
    def __init__(self, lr=0.05, epochs=200, l2=0.001):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w = None

    def _add_bias(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def _sigmoid(self, z):
        z = np.clip(z, -40, 40)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        Xb = self._add_bias(X)
        n, d = Xb.shape
        self.w = np.random.randn(d) * 0.01

        for ep in range(self.epochs):
            z = Xb @ self.w
            p = self._sigmoid(z)

            # gradient
            grad = (Xb.T @ (p - y)) / n
            grad[:-1] += self.l2 * self.w[:-1]

            self.w -= self.lr * grad

            if ep % 20 == 0:
                loss = -np.mean(y * np.log(p + 1e-12) +
                                (1 - y) * np.log(1 - p + 1e-12))
                print(f"Epoch {ep} | Loss {loss:.4f}")

    def predict_proba(self, X):
        Xb = self._add_bias(X)
        return self._sigmoid(Xb @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ------------------------------------------------------------
# EVERYTHING BELOW IS TRAINING CODE
# ------------------------------------------------------------
if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "Flight_Delay.parquet")
    CLUSTER_DIR = os.path.join(BASE_DIR, "ClusteringAlgorithm")

    # load scaler + encoders
    scaler = joblib.load(os.path.join(CLUSTER_DIR, "scaler.joblib"))
    enc_M = joblib.load(os.path.join(CLUSTER_DIR, "encoder_Marketing_Airline_Network.joblib"))
    enc_O = joblib.load(os.path.join(CLUSTER_DIR, "encoder_OriginCityName.joblib"))
    enc_D = joblib.load(os.path.join(CLUSTER_DIR, "encoder_DestCityName.joblib"))

    # use same 16 clustering features
    FEATURES_16 = [
        "Year", "DayofMonth", "FlightDate",
        "Marketing_Airline_Network", "OriginCityName", "DestCityName",
        "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes",
        "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn", "CRSArrTime",
        "Distance"
    ]
    TARGET = "ArrDelayMinutes"

    print("\nLoading dataset:", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)
    df = df[FEATURES_16 + [TARGET]].dropna()

    # Encode categoricals
    df["Marketing_Airline_Network"] = enc_M.transform(df["Marketing_Airline_Network"].astype(str))
    df["OriginCityName"] = enc_O.transform(df["OriginCityName"].astype(str))
    df["DestCityName"] = enc_D.transform(df["DestCityName"].astype(str))

    # convert flight date
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    df["FlightDate"] = df["FlightDate"].dt.dayofyear.fillna(0)

    # Build matrix
    X = df[FEATURES_16].values.astype(float)
    y = (df[TARGET] >= 15).astype(int).values

    # scale using clustering scaler
    X_scaled = scaler.transform(X)

    # train/test split
    np.random.seed(42)
    idx = np.arange(len(X_scaled))
    np.random.shuffle(idx)
    cut = int(0.8 * len(idx))
    train_idx, test_idx = idx[:cut], idx[cut:]

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # train classifier
    clf = ScratchLogisticRegression(lr=0.05, epochs=200, l2=0.001)
    clf.fit(X_train, y_train)

    # evaluate
    probs = clf.predict_proba(X_test)
    pred  = (probs >= 0.5).astype(int)

    acc = np.mean(pred == y_test)
    prec = np.sum((pred==1)&(y_test==1)) / max(1, np.sum(pred==1))
    rec  = np.sum((pred==1)&(y_test==1)) / max(1, np.sum(y_test==1))
    f1   = 0 if (prec+rec)==0 else 2*(prec*rec)/(prec+rec)

    print("\n=== PERFORMANCE ===")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)

    # save weights
    OUT = os.path.join(os.path.dirname(__file__), "logreg_weights.joblib")
    joblib.dump(clf.w, OUT)
    print("\nSaved weights to:", OUT)
