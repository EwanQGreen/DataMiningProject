import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_parquet('archive/Flight_Delay.parquet')

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# Save to CSV for inspection (optional)
df.to_csv('Flight_Delay.csv', index=False)

# Choose target and features (all exist in your dataset)
target = "ArrDelayMinutes"  # Predicting arrival delay in minutes
features = ["Month", "DayofMonth", "Distance", "DepTime", "CRSDepTime", "TaxiOut"]

# Drop missing values for simplicity
df = df[features + [target]].dropna()

# Extract X and y
X = df[features].values
y = df[target].values.reshape(-1, 1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add bias term (column of 1s)
X_bias = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_bias, y, test_size=0.2, random_state=42)

# Initialize parameters
theta = np.zeros((X_train.shape[1], 1))

# Hyperparameters
alpha = 0.01
epochs = 1000

# Gradient Descent
for i in range(epochs):
    predictions = X_train.dot(theta)
    errors = predictions - y_train
    gradients = (2 / len(X_train)) * X_train.T.dot(errors)
    theta -= alpha * gradients

    if i % 100 == 0:
        mse = np.mean(errors ** 2)
        print(f"Iteration {i}: MSE = {mse:.2f}")

# Evaluate on test set
y_pred = X_test.dot(theta)
mse_test = np.mean((y_pred - y_test) ** 2)
print(f"\nTest MSE: {mse_test:.2f}")

# Example predictions
print("\nSample predictions:")
for i in range(5):
    print(f"Predicted: {y_pred[i][0]:.2f}, Actual: {y_test[i][0]:.2f}")
