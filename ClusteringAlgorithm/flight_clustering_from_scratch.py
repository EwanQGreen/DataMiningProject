# flight_clustering_from_scratch.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import joblib
import os

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
df = pd.read_parquet(
    r"C:\Users\solar\OneDrive\Documents\DataMiningProject\Flight_Delay.parquet"
)
print("Columns:", df.columns.tolist())

# ------------------------------------------------------------
# 2. Select Relevant Features
# ------------------------------------------------------------
features = [
    'Year', 'DayofMonth', 'FlightDate',
    'Marketing_Airline_Network', 'OriginCityName', 'DestCityName',
    'CRSDepTime', 'DepTime', 'DepDelay', 'DepDelayMinutes',
    'TaxiOut', 'WheelsOff', 'WheelsOn', 'TaxiIn', 'CRSArrTime',
    'Distance'
]
df = df[features].copy()

# ------------------------------------------------------------
# 3. Basic Cleaning
# ------------------------------------------------------------
df = df.dropna(subset=['DepDelay', 'DepDelayMinutes', 'Distance'])

# FlightDate -> day of year
df['FlightDate'] = pd.to_datetime(df['FlightDate'], errors='coerce')
df['FlightDate'] = df['FlightDate'].dt.dayofyear

# ------------------------------------------------------------
# 4. Encode categorical features (on full dataset!)
# ------------------------------------------------------------
categorical_cols = ['Marketing_Airline_Network', 'OriginCityName', 'DestCityName']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(df[col].astype(str))  # fit on full dataset
    df[col] = le.transform(df[col].astype(str))
    encoders[col] = le

# Fill numeric NaNs with median
df = df.fillna(df.median(numeric_only=True))

# ------------------------------------------------------------
# 5. Feature Scaling
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Optional: sample 20,000 rows for clustering
rng = np.random.default_rng(seed=42)
idx = rng.choice(X_scaled.shape[0], 20_000, replace=False)
X_sample = X_scaled[idx]

# ------------------------------------------------------------
# 6. K-Means Clustering
# ------------------------------------------------------------
K = 6
kmeans = MiniBatchKMeans(n_clusters=K, batch_size=10_000, max_iter=100, random_state=42)
labels = kmeans.fit_predict(X_sample)

# Assign cluster labels back to the full dataframe
df['Cluster'] = -1
df.iloc[idx, df.columns.get_loc('Cluster')] = labels

# ------------------------------------------------------------
# 7. Cluster Analysis (average delay)
# ------------------------------------------------------------
cluster_means = df[df['Cluster'] != -1].groupby('Cluster')['DepDelayMinutes'].mean().to_dict()
print("Cluster means:", cluster_means)

# ------------------------------------------------------------
# 8. PCA for visualization (optional)
# ------------------------------------------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

# ------------------------------------------------------------
# 9. Save models for GUI
# ------------------------------------------------------------
MODEL_DIR = os.path.dirname(__file__)

# Save KMeans
joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.joblib"))

# Save scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

# Save encoders
for col, le in encoders.items():
    joblib.dump(le, os.path.join(MODEL_DIR, f"encoder_{col}.joblib"))

# Save cluster means
joblib.dump(cluster_means, os.path.join(MODEL_DIR, "cluster_means.joblib"))

# Save PCA
joblib.dump(pca, os.path.join(MODEL_DIR, "pca_model.joblib"))

print("All models saved to:", MODEL_DIR)
