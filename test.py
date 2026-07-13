from tensorflow.keras.models import load_model

model = load_model("models/lstm_model.keras")

print(model.summary())