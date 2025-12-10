import os
import numpy as np
import pandas as pd
import joblib

#paths
BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "Flight_Delay.parquet")
CLUST = os.path.join(BASE, "ClusteringAlgorithm")

#cols
FEAT = [
    "Year","DayofMonth","FlightDate",
    "Marketing_Airline_Network","OriginCityName","DestCityName",
    "CRSDepTime","DepTime","CRSArrTime","TaxiOut"
]
TGT = "ArrDelayMinutes"

#load preprocessing
scaler = joblib.load(os.path.join(CLUST,"scaler.joblib"))
enc_M = joblib.load(os.path.join(CLUST,"encoder_Marketing_Airline_Network.joblib"))
enc_O = joblib.load(os.path.join(CLUST,"encoder_OriginCityName.joblib"))
enc_D = joblib.load(os.path.join(CLUST,"encoder_DestCityName.joblib"))

#load data
df = pd.read_parquet(DATA)
df = df[FEAT+[TGT]].dropna()

#date
df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
df["FlightDate"] = df["FlightDate"].dt.dayofyear.fillna(0)

#encode
df["Marketing_Airline_Network"] = enc_M.transform(df["Marketing_Airline_Network"].astype(str))
df["OriginCityName"] = enc_O.transform(df["OriginCityName"].astype(str))
df["DestCityName"] = enc_D.transform(df["DestCityName"].astype(str))

#prep
X = df[FEAT].values.astype(float)
y = df[TGT].values.astype(float).reshape(-1,1)

#scale X
X_scaled = scaler.transform(X)
Xb = np.hstack([np.ones((X_scaled.shape[0],1)), X_scaled])

#train-test
np.random.seed(42)
idx = np.arange(len(Xb))
np.random.shuffle(idx)
cut = int(0.8*len(idx))
tr, te = idx[:cut], idx[cut:]
Xtr, Xte = Xb[tr], Xb[te]
ytr, yte = y[tr], y[te]

#scale y
y_mean = ytr.mean()
y_std = ytr.std()
ytr_s = (ytr - y_mean)/y_std

#train
theta = np.zeros(Xtr.shape[1])
alpha = 0.0001
epochs = 800

for i in range(epochs):
    preds = Xtr @ theta
    errs = preds.reshape(-1,1) - ytr_s
    grad = (2/len(Xtr)) * (Xtr.T @ errs).ravel()
    theta -= alpha * grad

    if i % 100 == 0:
        mse = np.mean(errs**2)
        print("Epoch", i, "MSE =", mse)

#predict
y_pred_s = (Xte @ theta).reshape(-1,1)
y_pred = y_pred_s * y_std + y_mean

mse_test = np.mean((y_pred - yte)**2)
print("Test MSE:", mse_test)

#save
OUT = os.path.join(os.path.dirname(__file__),"regression_theta.joblib")
meta = {"theta":theta, "y_mean":y_mean, "y_std":y_std}
joblib.dump(meta, OUT)
