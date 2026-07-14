# this is used to evaluate the metrics of the model and the predictions being made
# RMSE, MAE, MSE

"""
====================================================
Module : evaluate.py
Project: Airline Passenger Forecasting
Purpose: Evaluate LSTM Model Performance
====================================================
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from .predict import Predictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class Evaluator:
    """Evaluate model performance."""

    def __init__(self):
        pass

    def evaluate(self):

        # -----------------------------
        # Generate Predictions
        # -----------------------------
        predictor = Predictor()

        actual, predicted = predictor.predict()

        # -----------------------------
        # Calculate Metrics
        # -----------------------------
        mae = mean_absolute_error(
            actual,
            predicted
        )

        mse = mean_squared_error(
            actual,
            predicted
        )

        rmse = np.sqrt(mse)

        print("\nModel Evaluation")

        print(f"\nMAE  : {mae:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")

        # -----------------------------
        # Save Metrics
        # -----------------------------
        OUTPUTS_DIR.mkdir(exist_ok=True)

        with open(
            OUTPUTS_DIR / "metrics.txt",
            "w"
        ) as file:

            file.write("Model Evaluation\n")
            file.write("=====================\n")
            file.write(f"MAE  : {mae:.4f}\n")
            file.write(f"MSE  : {mse:.4f}\n")
            file.write(f"RMSE : {rmse:.4f}\n")

        print("\nMetrics Saved Successfully.")

        return mae, mse, rmse


if __name__ == "__main__":

    evaluator = Evaluator()

    mae, mse, rmse = evaluator.evaluate()