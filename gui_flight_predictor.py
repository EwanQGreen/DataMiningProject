# gui_flight_predictor.py

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)

# --- Clustering models ---
CLUSTER_DIR = os.path.join(BASE_DIR, "ClusteringAlgorithm")

kmeans_model = joblib.load(os.path.join(CLUSTER_DIR, "kmeans_model.joblib"))
scaler = joblib.load(os.path.join(CLUSTER_DIR, "scaler.joblib"))
pca = joblib.load(os.path.join(CLUSTER_DIR, "pca_model.joblib"))
cluster_means = joblib.load(os.path.join(CLUSTER_DIR, "cluster_means.joblib"))

encoders = {
    "OriginCityName": joblib.load(os.path.join(CLUSTER_DIR, "encoder_OriginCityName.joblib")),
    "DestCityName": joblib.load(os.path.join(CLUSTER_DIR, "encoder_DestCityName.joblib")),
    "Marketing_Airline_Network": joblib.load(os.path.join(CLUSTER_DIR, "encoder_Marketing_Airline_Network.joblib")),
}

# --- Logistic classifier ---
CLASSIFIER_DIR = os.path.join(BASE_DIR, "ClassifierAlgorithm")
logreg_weights = joblib.load(os.path.join(CLASSIFIER_DIR, "logreg_weights.joblib"))

# --- Regression model (manual gradient descent) ---
REG_DIR = os.path.join(BASE_DIR, "RegressionAlgorithm")

reg_theta = joblib.load(os.path.join(REG_DIR, "reg_theta.joblib"))
reg_mean = joblib.load(os.path.join(REG_DIR, "reg_mean.joblib"))
reg_std = joblib.load(os.path.join(REG_DIR, "reg_std.joblib"))

def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------
# Load full dataset for cluster summaries
# ------------------------------------------------------------
df_original = pd.read_parquet(
    r"C:\Users\solar\OneDrive\Documents\DataMiningProject\Flight_Delay.parquet"
)

features = [
    "Year", "DayofMonth", "FlightDate",
    "OriginCityName", "DestCityName", "Marketing_Airline_Network",
    "CRSDepTime", "DepTime", "CRSArrTime", "TaxiOut"
]

df_original = df_original[features].copy()

df_original["FlightDate"] = pd.to_datetime(df_original["FlightDate"], errors="coerce")
df_original["FlightDate"] = df_original["FlightDate"].dt.dayofyear

# Encode categories
for col, enc in encoders.items():
    df_original[col] = enc.transform(df_original[col].astype(str))

df_original = df_original.fillna(df_original.median(numeric_only=True))

# Scale
X_scaled_full = scaler.transform(df_original)

# Assign clusters
df_original["Cluster"] = kmeans_model.predict(X_scaled_full)

# ------------------------------------------------------------
# Build cluster summary table
# ------------------------------------------------------------
def top_values(series, n=3):
    return series.value_counts().head(n).index.tolist()

cluster_summary_df = df_original.groupby("Cluster").agg({
    "DepTime": ["mean", "std"],
    "CRSDepTime": ["mean", "std"],
    "TaxiOut": ["mean", "std"],
    "OriginCityName": lambda x: top_values(x),
    "DestCityName": lambda x: top_values(x),
    "Marketing_Airline_Network": lambda x: top_values(x)
})

cluster_summary_df.columns = [
    "DepTime_mean", "DepTime_std",
    "CRSDepTime_mean", "CRSDepTime_std",
    "TaxiOut_mean", "TaxiOut_std",
    "Top_Origins", "Top_Destinations", "Top_Airlines"
]


# ------------------------------------------------------------
# GUI
# ------------------------------------------------------------
class FlightPredictorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Flight Delay, Cluster & Logistic Predictor")
        master.geometry("700x900")

        self.inputs = {}
        self.features = features

        # Example prefill
        prefill = {
            "Year": "2020",
            "DayofMonth": "14",
            "FlightDate": "2020-07-14",
            "OriginCityName": "Dallas/Fort Worth, TX",
            "DestCityName": "Atlanta, GA",
            "Marketing_Airline_Network": "AA",
            "CRSDepTime": "1145",
            "DepTime": "1153",
            "CRSArrTime": "1425",
            "TaxiOut": "0"
        }

        for i, f in enumerate(self.features):
            ttk.Label(master, text=f).grid(row=i, column=0, sticky="w", padx=10, pady=5)
            e = ttk.Entry(master)
            e.grid(row=i, column=1, padx=10, pady=5)
            e.insert(0, prefill.get(f, ""))
            self.inputs[f] = e

        ttk.Button(master, text="Predict", command=self.predict).grid(
            row=len(self.features), column=0, columnspan=2, pady=20
        )

        self.output_text = tk.Text(master, height=35, width=80)
        self.output_text.grid(row=len(self.features)+1, column=0, columnspan=2)

    # ------------------------------------------------------------
    # Prediction Logic
    # ------------------------------------------------------------
    def predict(self):
        try:
            # Read inputs
            row = {}
            for f, e in self.inputs.items():
                v = e.get()
                row[f] = np.nan if v == "" else v

            df = pd.DataFrame([row])

            # Fix date
            df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
            df["FlightDate"] = df["FlightDate"].dt.dayofyear

            # Encode categorical
            for col, enc in encoders.items():
                df[col] = enc.transform(df[col].astype(str))

            df = df.fillna(df.median(numeric_only=True))

            # Scale
            X_scaled = scaler.transform(df)

            # ------------------ KMEANS ------------------
            cluster = kmeans_model.predict(X_scaled)[0]
            est_delay = cluster_means.get(cluster, "Unknown")

            summary = cluster_summary_df.loc[cluster]

            top_origins = encoders["OriginCityName"].inverse_transform(summary["Top_Origins"])
            top_dests = encoders["DestCityName"].inverse_transform(summary["Top_Destinations"])
            top_air = encoders["Marketing_Airline_Network"].inverse_transform(summary["Top_Airlines"])

            # ------------------ LOGISTIC ------------------
            Xb = np.hstack([X_scaled, np.ones((1, 1))])   # add bias
            prob = sigmoid(Xb @ logreg_weights)[0]
            binary = "YES" if prob >= 0.5 else "NO"

            # ------------------ REGRESSION (Manual Gradient Descent Model) ------------------
            # Extract required regression inputs
            reg_features = ["Year","DayofMonth","FlightDate","OriginCityName","DestCityName","Marketing_Airline_Network","CRSDepTime","DepTime","CRSArrTime","TaxiOut"]
            X_reg = df[reg_features].values.astype(float)

            # Manual standardization
            X_reg_scaled = (X_reg - reg_mean) / reg_std

            # Add bias term
            X_reg_bias = np.c_[np.ones((X_reg_scaled.shape[0], 1)), X_reg_scaled]

            # Predict delay
            reg_prediction = float(X_reg_bias.dot(reg_theta)[0])


            # ------------------ Output ------------------
            self.output_text.delete("1.0", tk.END)

            self.output_text.insert(tk.END, "=== KMEANS CLUSTERING ===\n")
            self.output_text.insert(tk.END, f"Cluster: {cluster}\n")
            self.output_text.insert(tk.END, f"Estimated Delay: {est_delay} minutes\n\n")

            self.output_text.insert(tk.END, "Cluster Characteristics:\n")
            self.output_text.insert(tk.END, f" Avg DepTime: {summary['DepTime_mean']:.1f} ± {summary['DepTime_std']:.1f}\n")
            self.output_text.insert(tk.END, f" Avg CRSDepTime: {summary['CRSDepTime_mean']:.1f} ± {summary['CRSDepTime_std']:.1f}\n")
            self.output_text.insert(tk.END, f" Avg TaxiOut: {summary['TaxiOut_mean']:.1f} ± {summary['TaxiOut_std']:.1f}\n")
            self.output_text.insert(tk.END, f" Top Origins: {list(top_origins)}\n")
            self.output_text.insert(tk.END, f" Top Destinations: {list(top_dests)}\n")
            self.output_text.insert(tk.END, f" Top Airlines: {list(top_air)}\n\n")

            self.output_text.insert(tk.END, "=== LOGISTIC DELAY CLASSIFIER ===\n")
            self.output_text.insert(tk.END, f"Probability of delay ≥ 15 minutes: {prob:.3f}\n")
            self.output_text.insert(tk.END, f"Delay? {binary}\n")

            self.output_text.insert(tk.END, "\n=== REGRESSION: DELAY IN MINUTES ===\n")
            self.output_text.insert(tk.END, f"Predicted Arrival Delay: {reg_prediction:.2f} minutes\n")


        except Exception as e:
            messagebox.showerror("Error", str(e))


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    gui = FlightPredictorGUI(root)
    root.mainloop()
