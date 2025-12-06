# gui_flight_predictor.py
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import joblib  # for saving/loading models
import os

# ------------------------------------------------------------
# Load the trained model and scaler
# ------------------------------------------------------------
# Make sure flight_clustering_from_scratch.py saves these after training:
# - kmeans model -> kmeans_model.joblib
# - scaler -> scaler.joblib
# - label encoders -> encoder_OriginCity.joblib, etc.
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'ClusteringAlgorithm')

kmeans_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
encoders = {
    "OriginCityName": joblib.load(os.path.join(MODEL_DIR, "encoder_OriginCityName.joblib")),
    "DestCityName": joblib.load(os.path.join(MODEL_DIR, "encoder_DestCityName.joblib")),
    "Marketing_Airline_Network": joblib.load(os.path.join(MODEL_DIR, "encoder_Marketing_Airline_Network.joblib")),
}

# Optional: load PCA for visualization
pca = joblib.load(os.path.join(MODEL_DIR, "pca_model.joblib"))

# ------------------------------------------------------------
# GUI
# ------------------------------------------------------------
class FlightPredictorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Flight Delay & Cluster Predictor")
        master.geometry("500x600")
        
        self.inputs = {}
        features = [
            "Year", "DayofMonth", "FlightDate", 
            "OriginCityName", "DestCityName", "Marketing_Airline_Network",
            "CRSDepTime", "DepTime", "CRSArrTime", "TaxiOut"
        ]
        
        for i, f in enumerate(features):
            label = ttk.Label(master, text=f)
            label.grid(row=i, column=0, sticky='w', padx=10, pady=5)
            entry = ttk.Entry(master)
            entry.grid(row=i, column=1, padx=10, pady=5)
            self.inputs[f] = entry
            
        self.predict_btn = ttk.Button(master, text="Predict Delay & Cluster", command=self.predict)
        self.predict_btn.grid(row=len(features), column=0, columnspan=2, pady=20)
        
        self.output_text = tk.Text(master, height=10, width=60)
        self.output_text.grid(row=len(features)+1, column=0, columnspan=2, padx=10, pady=10)
        
    def predict(self):
        try:
            # ------------------------------------------------------------
            # 1. Read user input
            # ------------------------------------------------------------
            data = {}
            for f, entry in self.inputs.items():
                value = entry.get()
                if value == "":
                    data[f] = np.nan
                else:
                    data[f] = value
            
            # ------------------------------------------------------------
            # 2. Preprocess
            # ------------------------------------------------------------
            df_input = pd.DataFrame([data])
            
            # FlightDate -> day of year
            df_input['FlightDate'] = pd.to_datetime(df_input['FlightDate'], errors='coerce')
            df_input['FlightDate'] = df_input['FlightDate'].dt.dayofyear
            
            # Encode categorical
            for col, enc in encoders.items():
                df_input[col] = enc.transform(df_input[col].astype(str))
            
            # Fill NaN with median (or zero)
            df_input = df_input.fillna(0)
            
            # Scale
            X_scaled = scaler.transform(df_input)
            
            # ------------------------------------------------------------
            # 3. Predict cluster
            # ------------------------------------------------------------
            cluster_label = kmeans_model.predict(X_scaled)[0]
            
            # ------------------------------------------------------------
            # 4. Approximate delay
            # (use the mean delay of that cluster if available)
            # ------------------------------------------------------------
            # This requires cluster means saved from your clustering script
            cluster_means = joblib.load(os.path.join(MODEL_DIR, "cluster_means.joblib"))
            predicted_delay = cluster_means.get(cluster_label, "Unknown")
            
            # ------------------------------------------------------------
            # 5. Display results
            # ------------------------------------------------------------
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"Predicted Cluster: {cluster_label}\n")
            self.output_text.insert(tk.END, f"Estimated Departure Delay (minutes): {predicted_delay}\n")
            self.output_text.insert(tk.END, f"Similar Clusters: {[cluster_label]}\n")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
        

# ------------------------------------------------------------
# Run GUI
# ------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    gui = FlightPredictorGUI(root)
    root.mainloop()
