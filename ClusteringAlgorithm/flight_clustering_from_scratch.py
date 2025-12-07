# flight_clustering_selected_features_train.py
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
    'OriginCityName', 'DestCityName', 'Marketing_Airline_Network',
    'CRSDepTime', 'DepTime', 'CRSArrTime', 'TaxiOut'
]
df = df[features].copy()

# ------------------------------------------------------------
# 3. Basic Cleaning
# ------------------------------------------------------------
# Drop rows with missing numeric values
df = df.dropna(subset=['DepTime', 'CRSDepTime', 'CRSArrTime', 'TaxiOut'])

# FlightDate -> day of year
df['FlightDate'] = pd.to_datetime(df['FlightDate'], errors='coerce')
df['FlightDate'] = df['FlightDate'].dt.dayofyear

# ------------------------------------------------------------
# 4. Encode categorical features
# ------------------------------------------------------------
categorical_cols = ['Marketing_Airline_Network', 'OriginCityName', 'DestCityName']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(df[col].astype(str))  # fit on full dataset
    df[col] = le.transform(df[col].astype(str))
    encoders[col] = le

# Fill remaining numeric NaNs with median
df = df.fillna(df.median(numeric_only=True))

# ------------------------------------------------------------
# 5. Feature Scaling
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ------------------------------------------------------------
# 6. K-Means Clustering
# ------------------------------------------------------------
K = 6
kmeans = MiniBatchKMeans(n_clusters=K, batch_size=10_000, max_iter=100, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Assign cluster labels
df['Cluster'] = labels

# ------------------------------------------------------------
# 7. Cluster Analysis (optional)
# ------------------------------------------------------------
cluster_means = df.groupby('Cluster')['TaxiOut'].mean().to_dict()
print("Cluster means (TaxiOut):", cluster_means)

# ------------------------------------------------------------
# 8. PCA for visualization (optional)
# ------------------------------------------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

# ------------------------------------------------------------
# 9. Save models
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

print("All models retrained and saved to:", MODEL_DIR)

