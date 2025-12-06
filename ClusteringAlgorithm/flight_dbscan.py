# flight_dbscan.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import kagglehub


# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
# Make sure to adjust the filename to your local file name
df = pd.read_parquet("Flight_Delay.parquet")
print(df.columns.tolist())
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
# Drop rows with missing critical numerical values
df = df.dropna(subset=['DepDelay', 'DepDelayMinutes', 'Distance'])

# Convert FlightDate to numerical (day of year)
df['FlightDate'] = pd.to_datetime(df['FlightDate'], errors='coerce')
df['FlightDate'] = df['FlightDate'].dt.dayofyear

# Encode categorical columns
categorical_cols = ['Marketing_Airline_Network', 'OriginCityName', 'DestCityName']
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

# Replace any remaining NaN with median
df = df.fillna(df.median(numeric_only=True))

# ------------------------------------------------------------
# 4. Feature Scaling
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ------------------------------------------------------------
# 5. DBSCAN Clustering
# ------------------------------------------------------------
# Epsilon and min_samples may need tuning depending on dataset size

# eps: roughly controls how close points must be to be a cluster
# min_samples: how many neighbors to form a cluster
dbscan = DBSCAN(eps=1.5, min_samples=10, n_jobs=-1)
labels = dbscan.fit_predict(X_scaled)

df['Cluster'] = labels

# ------------------------------------------------------------
# 6. Cluster Analysis
# ------------------------------------------------------------
# Count number of flights in each cluster
cluster_counts = df['Cluster'].value_counts().sort_index()
print("Cluster counts:")
print(cluster_counts)

# Show mean delay per cluster (excluding noise = -1)
valid_clusters = df[df['Cluster'] != -1].groupby('Cluster')['DepDelayMinutes'].mean()
print("\nAverage delay per cluster (minutes):")
print(valid_clusters)

# ------------------------------------------------------------
# 7. Optional: 2D Visualization using PCA
# ------------------------------------------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

plt.figure(figsize=(10, 7))
scatter = plt.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='tab10', s=10)
plt.title("DBSCAN Clusters of Flights (PCA projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(scatter, label="Cluster ID")
plt.show()
