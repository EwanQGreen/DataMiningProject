import pandas as pd
import numpy as np

# Load dataset
df = pd.read_parquet(
    r"C:\Users\solar\OneDrive\Documents\DataMiningProject\Flight_Delay.parquet"
)

# Columns of interest
target = "ArrDelayMinutes"
features = ["Month", "DayofMonth", "Distance", "DepTime", "CRSDepTime", "TaxiOut"]

# Drop rows with missing values
df = df[features + [target]].dropna()

# Extract X and y
X = df[features].values
y = df[target].values.reshape(-1, 1)

# -------------------------
# Manual Standardization
# -------------------------
# Compute mean and std for each feature
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

# Add bias term
X_bias = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

# -------------------------
# Manual train-test split
# -------------------------
np.random.seed(42)
indices = np.arange(X_bias.shape[0])
np.random.shuffle(indices)

split = int(0.8 * len(indices))
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X_bias[train_idx], X_bias[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# -------------------------
# Initialize parameters
# -------------------------
theta = np.zeros((X_train.shape[1], 1))

# Hyperparameters
alpha = 0.01
epochs = 1000

# -------------------------
# Gradient Descent
# -------------------------
for i in range(epochs):
    predictions = X_train.dot(theta)
    errors = predictions - y_train
    gradients = (2 / len(X_train)) * X_train.T.dot(errors)
    theta -= alpha * gradients

    if i % 100 == 0:
        mse = np.mean(errors ** 2)
        print(f"Iteration {i}: MSE = {mse:.2f}")

# -------------------------
# Evaluate on test set
# -------------------------
y_pred = X_test.dot(theta)
mse_test = np.mean((y_pred - y_test) ** 2)
print(f"\nTest MSE: {mse_test:.2f}")

# Sample predictions
print("\nSample predictions:")
for i in range(5):
    print(f"Predicted: {y_pred[i][0]:.2f}, Actual: {y_test[i][0]:.2f}")
