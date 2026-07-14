# this file is used to predict the future values of the airline passengers dataset using the trained model
# Future forecasting
 
"""
====================================================
Module : forecast.py
Project: Airline Passenger Forecasting
Purpose: Forecast Future Passenger Counts
====================================================
"""
 
from pathlib import Path

import numpy as np
import joblib
 
from tensorflow.keras.models import load_model
 
from .data_loader import DataLoader
from .preprocessing import Preprocessor
 
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class Forecaster:

    def __init__(self):

        self.data_path = PROJECT_ROOT / "data" / "airline-passengers.csv"

        self.model_paths = [
            MODELS_DIR / "lstm_model.keras",
            MODELS_DIR / "lstm_model.h5",
        ]

        self.scaler_path = MODELS_DIR / "scaler.pkl"
        self.sequence_length = 12

    def forecast(self, future_months=12):
 
        # --------------------------
        # Load Dataset
        # --------------------------
 
        loader = DataLoader(self.data_path)
 
        df = loader.load_data()
 
        # --------------------------
        # Scale Dataset
        # --------------------------
 
        preprocessor = Preprocessor()
 
        scaled_df = preprocessor.scale_data(df)
 
        # --------------------------
        # Load Model
        # --------------------------
        model_path = None
        for candidate in self.model_paths:
            if candidate.exists():
                model_path = candidate
                break

        if model_path is None:
            raise FileNotFoundError(f"No trained model found in {MODELS_DIR}")

        model = load_model(str(model_path), compile=False)
 
        # --------------------------
        # Load Scaler
        # --------------------------
 
        scaler = joblib.load(str(self.scaler_path))
 
        # --------------------------
        # Last 12 Months
        # --------------------------
 
        last_sequence = scaled_df.values[-self.sequence_length:]
 
        future_predictions = []
 
        # --------------------------
        # Forecast Loop
        # --------------------------
 
        for _ in range(future_months):
 
            input_data = last_sequence.reshape(
                1,
                self.sequence_length,
                1
            )
 
            prediction = model.predict(
                input_data,
                verbose=0
            )
 
            future_predictions.append(prediction[0,0])
 
            last_sequence = np.vstack(
                (
                    last_sequence[1:],
                    prediction
                )
            )
 
        # --------------------------
        # Convert Back
        # --------------------------
 
        future_predictions = np.array(
            future_predictions
        ).reshape(-1,1)
 
        future_predictions = scaler.inverse_transform(
            future_predictions
        )
 
        print("\nFuture Forecast Completed.")
 
        return future_predictions
if __name__ == "__main__":
 
    forecaster = Forecaster()
 
    future = forecaster.forecast(future_months=12)
 
    print("\nNext 12 Month Forecast\n")
 
    for i, value in enumerate(future, start=1):
 
        print(f"Month {i} : {value[0]:.2f}")