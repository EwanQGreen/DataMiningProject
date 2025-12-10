import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
import os
from ClusteringAlgorithm.clustering import MiniBatchKMeansScratch


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR

CLUSTER_DIR = os.path.join(PROJECT_ROOT, "ClusteringAlgorithm")
CLASSIFIER_DIR = os.path.join(PROJECT_ROOT, "ClassifierAlgorithm")
REGRESSION_DIR = os.path.join(PROJECT_ROOT, "RegressionAlgorithm")
DATA_PATH = os.path.join(PROJECT_ROOT, "Flight_Delay.parquet")

# ------------------------------------------------------------
# Load Allowed Models
# ------------------------------------------------------------
kmeans_model = joblib.load(os.path.join(CLUSTER_DIR, "kmeans_custom.joblib"))
scaler = joblib.load(os.path.join(CLUSTER_DIR, "scaler.joblib"))

encoders = {
    "OriginCityName": joblib.load(os.path.join(CLUSTER_DIR, "encoder_OriginCityName.joblib")),
    "DestCityName": joblib.load(os.path.join(CLUSTER_DIR, "encoder_DestCityName.joblib")),
    "Marketing_Airline_Network": joblib.load(os.path.join(CLUSTER_DIR, "encoder_Marketing_Airline_Network.joblib")),
}

logreg_weights = joblib.load(os.path.join(CLASSIFIER_DIR, "logreg_weights.joblib"))

# Load regression model (supports dict or vector)
reg_obj = joblib.load(os.path.join(REGRESSION_DIR, "regression_theta.joblib"))
if isinstance(reg_obj, dict):
    reg_theta = np.asarray(reg_obj["theta"])
    y_mean = float(reg_obj["y_mean"])
    y_std = float(reg_obj["y_std"])
else:
    reg_theta = np.asarray(reg_obj)
    y_mean = 0.0
    y_std = 1.0

def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1 / (1 + np.exp(-z))

# ------------------------------------------------------------
# Features
# ------------------------------------------------------------
FEATURES = [
    "Year",
    "DayofMonth",
    "FlightDate",
    "Marketing_Airline_Network",
    "OriginCityName",
    "DestCityName",
    "CRSDepTime",
    "DepTime",
    "CRSArrTime",
    "TaxiOut",
]

visible_features = FEATURES[:]

# ------------------------------------------------------------
# Load original dataset for cluster profiling
# ------------------------------------------------------------
df_original = pd.read_parquet(DATA_PATH)
df_original = df_original[FEATURES + ["ArrDelayMinutes"]].dropna()

df_original["FlightDate"] = pd.to_datetime(df_original["FlightDate"], errors="coerce")
df_original["FlightDate"] = df_original["FlightDate"].dt.dayofyear.fillna(0)

for col, enc in encoders.items():
    df_original[col] = enc.transform(df_original[col].astype(str))

df_original = df_original.fillna(df_original.median(numeric_only=True))

X_full = df_original[FEATURES].values.astype(float)
X_scaled_full = scaler.transform(X_full)
df_original["Cluster"] = kmeans_model.predict(X_scaled_full)

def top_values(series, n=3):
    return list(series.value_counts().head(n).index)

# ----------------- CLUSTER SUMMARY -----------------
cluster_summary_df = df_original.groupby("Cluster").agg({
    "ArrDelayMinutes": "mean",
    "DepTime": ["mean", "std"],
    "CRSDepTime": ["mean", "std"],
    "TaxiOut": ["mean", "std"],
    "OriginCityName": lambda x: top_values(x),
    "DestCityName": lambda x: top_values(x),
    "Marketing_Airline_Network": lambda x: top_values(x),
})

cluster_summary_df.columns = [
    "ClusterDelayMean",
    "DepTime_mean", "DepTime_std",
    "CRSDepTime_mean", "CRSDepTime_std",
    "TaxiOut_mean", "TaxiOut_std",
    "Top_Origins", "Top_Destinations", "Top_Airlines",
]

# ------------------------------------------------------------
# GUI
# ------------------------------------------------------------
class FlightPredictorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Flight Delay Predictor")
        master.geometry("950x900")

        # Load Azure theme
        master.tk.call("source", os.path.join(PROJECT_ROOT, "azure.tcl"))
        master.tk.call("set_theme", "dark")

        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)

        theme_btn = ttk.Button(
            master, text="Toggle Theme", command=self.toggle_theme
        )
        theme_btn.place(relx=0.95, rely=0.02, anchor="ne")

        container = ttk.Frame(master, padding=20)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)

        title_label = ttk.Label(
            container,
            text="✈ Flight Delay Predictor",
            font=("Segoe UI", 20, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 20))

        # ----------------- INPUTS -----------------
        input_frame = ttk.LabelFrame(container, text="Flight Information", padding=15)
        input_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        input_frame.columnconfigure(1, weight=1)

        self.inputs = {}

        example = {
            "Year": "2020",
            "DayofMonth": "14",
            "FlightDate": "2020-07-14",
            "OriginCityName": "Dallas/Fort Worth, TX",
            "DestCityName": "Atlanta, GA",
            "Marketing_Airline_Network": "AA",
            "CRSDepTime": "1145",
            "DepTime": "1153",
            "CRSArrTime": "1425",
            "TaxiOut": "10",
        }

        for i, f in enumerate(visible_features):
            ttk.Label(input_frame, text=f + ":").grid(
                row=i, column=0, sticky="e", padx=10, pady=5
            )
            entry = ttk.Entry(input_frame, width=30)
            entry.grid(row=i, column=1, sticky="ew", padx=10, pady=5)
            entry.insert(0, example.get(f, ""))
            self.inputs[f] = entry

        # Predict button
        predict_btn = ttk.Button(container, text="Predict", command=self.predict)
        predict_btn.grid(row=2, column=0, pady=20)

        # ----------------- OUTPUT -----------------
        output_frame = ttk.LabelFrame(container, text="Prediction Output", padding=10)
        output_frame.grid(row=3, column=0, sticky="nsew")
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        self.output_text = tk.Text(
            output_frame,
            wrap="word",
            height=30,
            font=("Consolas", 10),
            background="#1e1e1e",
            foreground="white",
            insertbackground="white",
            relief="flat",
            padx=10,
            pady=10
        )
        self.output_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            output_frame, orient="vertical", command=self.output_text.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.output_text.config(yscrollcommand=scrollbar.set)

        self.dark_mode = True

    # --------------------------------------------------------
    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.master.tk.call("set_theme", "dark" if self.dark_mode else "light")

    # --------------------------------------------------------
    def predict(self):
        try:
            row = {f: (widget.get() or np.nan) for f, widget in self.inputs.items()}
            df = pd.DataFrame([row])

            df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
            df["FlightDate"] = df["FlightDate"].dt.dayofyear.fillna(0)

            for col, enc in encoders.items():
                df[col] = enc.transform(df[col].astype(str))

            numeric_cols = [
                "Year", "DayofMonth", "FlightDate",
                "CRSDepTime", "DepTime", "CRSArrTime", "TaxiOut"
            ]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            df = df.fillna(0)

            X = df[FEATURES].values.astype(float)
            X_scaled = scaler.transform(X)

            # -------------------- K-MEANS --------------------
            cluster = kmeans_model.predict(X_scaled)[0]
            summary = cluster_summary_df.loc[cluster]

            top_orig = encoders["OriginCityName"].inverse_transform(summary["Top_Origins"])
            top_dest = encoders["DestCityName"].inverse_transform(summary["Top_Destinations"])
            top_air = encoders["Marketing_Airline_Network"].inverse_transform(summary["Top_Airlines"])

            # -------------------- LOGISTIC --------------------
            Xb = np.hstack([X_scaled, [[1]]])
            prob = sigmoid(Xb @ logreg_weights)[0]
            delayed = "YES" if prob >= 0.5 else "NO"

            # -------------------- REGRESSION --------------------
            y_scaled_pred = float(Xb @ reg_theta)
            reg_delay = y_scaled_pred * y_std + y_mean

            # ---------------------------------------------------
            # OUTPUT
            # ---------------------------------------------------
            self.output_text.delete("1.0", tk.END)

            self.output_text.insert(tk.END, "=== KMEANS CLUSTERING ===\n")
            self.output_text.insert(tk.END, f"Cluster: {cluster}\n")
            self.output_text.insert(tk.END, f"Estimated Delay (cluster mean): {summary['ClusterDelayMean']:.1f} min\n\n")

            self.output_text.insert(tk.END, "Cluster Characteristics:\n")
            self.output_text.insert(tk.END, f" Avg DepTime: {summary['DepTime_mean']:.1f} ± {summary['DepTime_std']:.1f}\n")
            self.output_text.insert(tk.END, f" Avg CRSDepTime: {summary['CRSDepTime_mean']:.1f} ± {summary['CRSDepTime_std']:.1f}\n")
            self.output_text.insert(tk.END, f" Avg TaxiOut: {summary['TaxiOut_mean']:.1f} ± {summary['TaxiOut_std']:.1f}\n")
            self.output_text.insert(tk.END, f" Top Origins: {list(top_orig)}\n")
            self.output_text.insert(tk.END, f" Top Destinations: {list(top_dest)}\n")
            self.output_text.insert(tk.END, f" Top Airlines: {list(top_air)}\n\n")

            self.output_text.insert(tk.END, "=== LOGISTIC REGRESSION (Delay ≥ 15 minutes) ===\n")
            self.output_text.insert(tk.END, f"Probability Delay ≥ 15 min: {prob:.3f}\n")
            self.output_text.insert(tk.END, f"Delayed? {delayed}\n\n")

            self.output_text.insert(tk.END, "=== LINEAR REGRESSION (Predicted Delay in Minutes) ===\n")
            self.output_text.insert(tk.END, f"Predicted Arrival Delay: {reg_delay:.2f} minutes\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))

# ------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    gui = FlightPredictorGUI(root)
    root.mainloop()

