# this file is used to load the data from the dataset airline-passengers.csv
# Load dataset
 
"""
====================================================
Module : data_loader.py
Project: Airline Passenger Forecasting
Purpose: Load and validate the dataset
====================================================
"""
 
# Import required libraries
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DataLoader:
    """
    A class responsible for loading the time series dataset.
    """
 
    def __init__(self, file_path):
        """
        Constructor
 
        Parameters
        ----------
        file_path : str
            Path of the CSV dataset.
        """
 
        self.file_path = Path(file_path)
        if not self.file_path.is_absolute():
            self.file_path = PROJECT_ROOT / self.file_path
 
    def load_data(self):
        """
        Load the dataset and return a DataFrame.
 
        Returns
        -------
        pandas.DataFrame
        """
 
        # -----------------------------
        # Step 1 : Check file existence
        # -----------------------------
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at:\n{self.file_path}"
            )
 
        print("Dataset Found Successfully.\n")
 
        # -----------------------------
        # Step 2 : Read CSV
        # -----------------------------
        
        try:
            df = pd.read_csv(self.file_path)
        except pd.errors.EmptyDataError:
            raise ValueError("Dataset is empty.")
        except pd.errors.ParserError:
            raise ValueError("Invalid CSV format.")
        except Exception as e:
            raise Exception(f"Error reading dataset: {e}")

        print("Dataset Loaded Successfully.\n")
 
        # -----------------------------
        # Step 3 : Display first rows
        # -----------------------------
        print("First 5 Rows")
        print(df.head())
 
        # -----------------------------
        # Step 4 : Dataset Shape
        # -----------------------------
        print("\nDataset Shape")
        print(df.shape)
 
        # -----------------------------
        # Step 5 : Dataset Information
        # -----------------------------
        print("\nDataset Information")
        print(df.info())
 
        # -----------------------------
        # Step 6 : Missing Values
        # -----------------------------
        print("\nMissing Values")
        print(df.isnull().sum())
 
        # -----------------------------
        # Step 7 : Convert Month column
        # -----------------------------
        if "month" not in df.columns:
            raise ValueError("'month' column not found.")
        print("\nConverting 'month' column to Datetime format.")
        
        try:
            df["month"] = pd.to_datetime(df["month"])
        except Exception:
            raise ValueError("Invalid values found in 'month' column.")
        
        print("\nData Types After Conversion")
        print(df.dtypes)
 
        # -----------------------------
        # Step 8 : Set Month as Index
        # -----------------------------
        df.set_index("month", inplace=True)
 
        print("\nMonth column converted into Datetime.")
 
        print("Month column set as Index.\n")
 
        # -----------------------------
        # Step 9 : Return DataFrame
        # -----------------------------
        return df
   
 
# ===========================================
# Test the module
# ===========================================
 
if __name__ == "__main__":
 
    DATA_PATH = "data/airline-passengers.csv"
 
    loader = DataLoader(DATA_PATH)
 
    df = loader.load_data()
 
    print("\nProcessed Dataset")
    print(df.head())
 