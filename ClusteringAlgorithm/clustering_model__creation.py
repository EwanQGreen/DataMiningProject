import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from ClusteringAlgorithm.clustering import MiniBatchKMeansScratch


#load dataset
df = pd.read_parquet(
    r"C:\Users\solar\OneDrive\Documents\DataMiningProject\Flight_Delay.parquet"
)
print("Columns:", df.columns.tolist())

#features
features = [
    'Year', 'DayofMonth', 'FlightDate',
    'OriginCityName', 'DestCityName', 'Marketing_Airline_Network',
    'CRSDepTime', 'DepTime', 'CRSArrTime', 'TaxiOut'
]
df = df[features].copy()

#clean
df = df.dropna(subset=['DepTime', 'CRSDepTime', 'CRSArrTime', 'TaxiOut'])

df['FlightDate'] = pd.to_datetime(df['FlightDate'], errors='coerce')
df['FlightDate'] = df['FlightDate'].dt.dayofyear

#encode
categorical_cols = ['Marketing_Airline_Network', 'OriginCityName', 'DestCityName']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

df = df.fillna(df.median(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

K = 6
kmeans = MiniBatchKMeansScratch(n_clusters=K, max_iter=50, random_state=42)
kmeans.fit(X_scaled)

df['Cluster'] = kmeans.predict(X_scaled)

cluster_sizes = df["Cluster"].value_counts().sort_index().to_dict()
print("Cluster sizes:", cluster_sizes)

#Save preprocessing
MODEL_DIR = os.path.dirname(__file__)

joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_custom.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

for col, le in encoders.items():
    joblib.dump(le, os.path.join(MODEL_DIR, f"encoder_{col}.joblib"))

print("Models saved to:", MODEL_DIR)
