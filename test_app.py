import streamlit as st
from tensorflow.keras.models import load_model
from pathlib import Path

st.title("Model Load Test")

model_path = Path("models/lstm_model.keras")

st.write("Exists:", model_path.exists())
st.write("Size:", model_path.stat().st_size)

if st.button("Load Model"):
    try:
        model = load_model(model_path, compile=False)
        st.success("Model loaded!")
        model.summary(print_fn=st.text)
    except Exception as e:
        st.exception(e)