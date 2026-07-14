# this file is used to store the predictions being made using the model
# Predictions
 
"""
====================================================
Module : predict.py
Project: Airline Passenger Forecasting
Purpose: Predict Passenger Counts
====================================================
"""
 
from pathlib import Path

import joblib
import numpy as np

from tensorflow.keras.models import load_model
 
from .data_loader import DataLoader
from .preprocessing import Preprocessor
from .sequence_generator import SequenceGenerator
from .train_test_split import TimeSeriesSplit
 
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class Predictor:

    def __init__(self):

        self.data_path = PROJECT_ROOT / "data" / "airline-passengers.csv"

        self.model_paths = [
            MODELS_DIR / "lstm_model.keras",
            MODELS_DIR / "lstm_model.h5",
        ]

        self.scaler_path = MODELS_DIR / "scaler.pkl"

    def predict(self):
 
        # ----------------------------
        # Load Dataset
        # ----------------------------
 
        loader = DataLoader(self.data_path)
 
        df = loader.load_data()
 
        # ----------------------------
        # Scale Dataset
        # ----------------------------
 
        preprocessor = Preprocessor()
 
        scaled_df = preprocessor.scale_data(df)
 
        # ----------------------------
        # Generate Sequences
        # ----------------------------
 
        generator = SequenceGenerator(sequence_length=12)
 
        X, y = generator.create_sequences(scaled_df)
 
        # ----------------------------
        # Train Test Split
        # ----------------------------
 
        splitter = TimeSeriesSplit(train_size=0.80)
 
        X_train, X_test, y_train, y_test = splitter.split(X, y)
 
        # ----------------------------
        # Load Model
        # ----------------------------
        model_path = None
        for candidate in self.model_paths:
            if candidate.exists():
                model_path = candidate
                break

        if model_path is None:
            raise FileNotFoundError(f"No trained model found in {MODELS_DIR}")

        model = load_model(str(model_path), compile=False)
 
        print("\nModel Loaded Successfully.")
 
        # ----------------------------
        # Load Scaler
        # ----------------------------
 
        scaler = joblib.load(str(self.scaler_path))
 
        print("Scaler Loaded Successfully.")
 
        # ----------------------------
        # Predict
        # ----------------------------
 
        predictions = model.predict(X_test)
 
        # ----------------------------
        # Convert back to original scale
        # ----------------------------
 
        predictions = scaler.inverse_transform(predictions)
 
        y_test = scaler.inverse_transform(y_test)
 
        print("\nPrediction Completed.")
 
        return y_test, predictions
if __name__ == "__main__":
 
    predictor = Predictor()
 
    actual, predicted = predictor.predict()
 
    print("\nFirst 10 Predictions\n")
 
    for i in range(10):
 
        print(
            f"Actual : {actual[i][0]:.2f}   "
            f"Predicted : {predicted[i][0]:.2f}"
        )