# Airline Passenger Forecasting

This project builds a Streamlit web app for forecasting airline passenger traffic using a recurrent neural network (RNN) approach. It loads historical passenger data, visualizes trends, and allows users to generate future passenger forecasts for a selected number of months.

## Features
- Upload or load historical airline passenger data
- Visualize historical trends with interactive charts
- Train and evaluate an LSTM-based forecasting model
- Generate future passenger predictions
- Download forecast results as CSV

## Tech Stack
- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn
- TensorFlow / Keras
- Joblib

## Project Structure
- app.py: Streamlit web application
- src/: core modules for loading data, preprocessing, sequence generation, training, forecasting, and evaluation
- data/: dataset files
- models/: trained model and scaler files
- assets/: images and other static assets

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- A trained model file is required for forecasting.
- The app is designed for time-series forecasting and demonstrates how sequence-based neural networks can be used for business forecasting tasks.
