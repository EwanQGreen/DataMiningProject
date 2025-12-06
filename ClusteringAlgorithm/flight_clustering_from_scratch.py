# flight_kmeans.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
df = pd.read_parquet(
    r"C:\Users\solar\OneDrive\Documents\DataMiningProject\ClusteringAlgorithm\Flight_Delay.parquet"
)
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
df = df.dropna(subset=['DepDelay', 'DepDelayMinutes', 'Distance'])

df['FlightDate'] = pd.to_datetime(df['FlightDate'], errors='coerce')
df['FlightDate'] = df['FlightDate'].dt.dayofyear

categorical_cols = ['Marketing_Airline_Network', 'OriginCityName', 'DestCityName']
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

df = df.fillna(df.median(numeric_only=True))

# ------------------------------------------------------------
# 4. Feature Scaling
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Optional sampling (200,000 rows)
rng = np.random.default_rng(seed=42)
idx = rng.choice(X_scaled.shape[0], 20_000, replace=False)
X_sample = X_scaled[idx]

# ------------------------------------------------------------
# 5. K-Means Clustering
# ------------------------------------------------------------
# Choose a number of clusters (you can change this)
K = 6

kmeans = MiniBatchKMeans(
    n_clusters=K,
    batch_size=10_000,
    max_iter=100,
    random_state=42
)

labels = kmeans.fit_predict(X_sample)

# Sample 200k rows for clustering
rng = np.random.default_rng(seed=42)
idx = rng.choice(X_scaled.shape[0], 20_000, replace=False)
X_sample = X_scaled[idx]

# K-means clustering
K = 6
kmeans = MiniBatchKMeans(n_clusters=K, batch_size=10_000, max_iter=100, random_state=42)
labels = kmeans.fit_predict(X_sample)

# Assign cluster labels correctly
df['Cluster'] = -1
df.iloc[idx, df.columns.get_loc('Cluster')] = labels


# ------------------------------------------------------------
# 6. Cluster Analysis
# ------------------------------------------------------------
cluster_counts = df['Cluster'].value_counts().sort_index()
print("Cluster counts:")
print(cluster_counts)

cluster_means = df[df['Cluster'] != -1].groupby('Cluster')['DepDelayMinutes'].mean()
print("\nAverage delay per cluster (minutes):")
print(cluster_means)

# ------------------------------------------------------------
# 7. PCA Visualization
# ------------------------------------------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    df['PCA1'], df['PCA2'], 
    c=df['Cluster'], cmap='tab10', s=8
)
plt.title("K-Means Clusters (PCA projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(scatter, label="Cluster ID")
plt.show()
