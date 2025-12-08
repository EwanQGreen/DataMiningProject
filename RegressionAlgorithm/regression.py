import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
#                 CONFIGURATION (DO NOT CHANGE)
# ============================================================

# SAME ORDER AS THE GUI
features = [
    "Year",
    "DayofMonth",
    "FlightDate",
    "OriginCityName",
    "DestCityName",
    "Marketing_Airline_Network",
    "CRSDepTime",
    "DepTime",
    "CRSArrTime",
    "TaxiOut"
]

target = "ArrDelayMinutes"

# Paths (same structure as your GUI)
BASE_DIR = os.path.dirname(__file__)
CLUSTER_DIR = os.path.join(BASE_DIR, "..", "ClusteringAlgorithm")

# Load all encoders used in GUI
enc_origin = joblib.load(os.path.join(CLUSTER_DIR, "encoder_OriginCityName.joblib"))
enc_dest = joblib.load(os.path.join(CLUSTER_DIR, "encoder_DestCityName.joblib"))
enc_air = joblib.load(os.path.join(CLUSTER_DIR, "encoder_Marketing_Airline_Network.joblib"))

# ============================================================
#                 LOAD DATASET
# ============================================================

df = pd.read_parquet(
    r"C:\Users\solar\OneDrive\Documents\DataMiningProject\Flight_Delay.parquet"
)

df = df[features + [target]].dropna()

# Convert FlightDate → Day of year
df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
df["FlightDate"] = df["FlightDate"].dt.dayofyear

# ============================================================
#        APPLY EXACT SAME ENCODING AS USED IN THE GUI
# ============================================================

df["OriginCityName"] = enc_origin.transform(df["OriginCityName"].astype(str))
df["DestCityName"] = enc_dest.transform(df["DestCityName"].astype(str))
df["Marketing_Airline_Network"] = enc_air.transform(df["Marketing_Airline_Network"].astype(str))

# ============================================================
#      SPLIT NUMERICAL VS CATEGORICAL FOR PROPER SCALING
# ============================================================

# Only these are numeric and safe to standardize
numeric_cols = ["Year", "DayofMonth", "FlightDate",
                "CRSDepTime", "DepTime", "CRSArrTime", "TaxiOut"]

categorical_cols = ["OriginCityName", "DestCityName", "Marketing_Airline_Network"]

# Extract matrices
X_num = df[numeric_cols].values
X_cat = df[categorical_cols].values.astype(float)
y = df[target].values.reshape(-1, 1)

# Standardize ONLY numeric features
X_mean = X_num.mean(axis=0)
X_std = X_num.std(axis=0)
X_num_scaled = (X_num - X_mean) / X_std

# Concatenate back in correct order matching GUI
X_scaled = np.concatenate([X_num_scaled, X_cat], axis=1)

# Add bias
X_bias = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

# ============================================================
#             TRAIN–TEST SPLIT + GRADIENT DESCENT
# ============================================================

np.random.seed(42)
indices = np.arange(len(X_bias))
np.random.shuffle(indices)

split = int(0.8 * len(indices))

train_idx = indices[:split]
test_idx = indices[split:]

X_train, X_test = X_bias[train_idx], X_bias[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

theta = np.zeros((X_train.shape[1], 1))

alpha = 0.01      # Learning rate
epochs = 1000     # Iterations

for i in range(epochs):
    predictions = X_train.dot(theta)
    errors = predictions - y_train
    gradients = (2 / len(X_train)) * X_train.T.dot(errors)
    theta -= alpha * gradients

# ============================================================
#                   SAVE TRAINED PARAMETERS
# ============================================================

save_path = os.path.dirname(__file__)

joblib.dump(theta, os.path.join(save_path, "reg_theta.joblib"))
joblib.dump(X_mean, os.path.join(save_path, "reg_mean.joblib"))
joblib.dump(X_std, os.path.join(save_path, "reg_std.joblib"))

# Also store numeric & categorical column ordering
joblib.dump(numeric_cols, os.path.join(save_path, "reg_numeric_cols.joblib"))
joblib.dump(categorical_cols, os.path.join(save_path, "reg_categorical_cols.joblib"))

print("\nSaved regression model successfully!")
print("Files created:")
print(" - reg_theta.joblib")
print(" - reg_mean.joblib")
print(" - reg_std.joblib")
print(" - reg_numeric_cols.joblib")
print(" - reg_categorical_cols.joblib")
