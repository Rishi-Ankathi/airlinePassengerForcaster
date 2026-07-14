# this is the file where the preprocessing of the data takes place
# Scaling & preprocessing
 
"""
====================================================
Module : preprocessing.py
Project: Airline Passenger Forecasting
Purpose: Scale the dataset using MinMaxScaler
====================================================
"""
 
# Import required libraries
from pathlib import Path

import joblib
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
 
 
class Preprocessor:
    """
    Preprocess the time series dataset.
    """
 
    def __init__(self):
        """
        Initialize the scaler.
        """
 
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler_path = MODELS_DIR / "scaler.pkl"
 
    def scale_data(self, df):
        """
        Scale the Passengers column.
 
        Parameters
        ----------
        df : pandas.DataFrame
 
        Returns
        -------
        scaled_df : pandas.DataFrame
        """
 
        print("\nOriginal Data")
        print(df.head())
 
        # Scale the Passengers column
        try:
            scaled_values = self.scaler.fit_transform(df[["passengers"]])
        except Exception as e:
            raise RuntimeError(f"Error scaling data: {e}")
 
        # Convert to DataFrame
        scaled_df = pd.DataFrame(
            scaled_values,
            columns=["passengers"],
            index=df.index
        )
 
        # print("\nScaled Data")
        # print(scaled_df.head())
 
        # Save the scaler
        try:
            MODELS_DIR.mkdir(exist_ok=True)
            joblib.dump(self.scaler, self.scaler_path)
        except Exception as e:
            raise RuntimeError(f"Unable to save scaler: {e}")
 
        print("\nScaler saved successfully.")
 
        return scaled_df
 
if __name__ == "__main__":
 
    from .data_loader import DataLoader
 
    DATA_PATH = "data/airline-passengers.csv"
 
    # Load data
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
 
    # Scale data
    preprocessor = Preprocessor()
 
    scaled_df = preprocessor.scale_data(df)
 
    print("\nScaled Dataset")
    print(scaled_df.head())
 