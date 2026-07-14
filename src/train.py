# this file is used to train the model using the training data and then save the trained model in the model.py file
"""
====================================================
Module : train.py
Project: Airline Passenger Forecasting
Purpose: Train the LSTM Model
====================================================
"""
 
from pathlib import Path

from .data_loader import DataLoader
from .preprocessing import Preprocessor
from .sequence_generator import SequenceGenerator
from .train_test_split import TimeSeriesSplit
from .model import ModelBuilder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "airline-passengers.csv"
MODELS_DIR = PROJECT_ROOT / "models"
 
 
class ModelTrainer:
 
    def __init__(self):
 
        self.data_path = DATA_PATH
 
    def train(self):
 
        # ----------------------------
        # Step 1 : Load Dataset
        # ----------------------------
 
        loader = DataLoader(self.data_path)
 
        df = loader.load_data()
 
        # ----------------------------
        # Step 2 : Preprocess
        # ----------------------------
 
        preprocessor = Preprocessor()
 
        scaled_df = preprocessor.scale_data(df)
 
        # ----------------------------
        # Step 3 : Generate Sequences
        # ----------------------------
 
        generator = SequenceGenerator(sequence_length=12)
 
        X, y = generator.create_sequences(scaled_df)
 
        # ----------------------------
        # Step 4 : Train Test Split
        # ----------------------------
 
        splitter = TimeSeriesSplit(train_size=0.80)
 
        X_train, X_test, y_train, y_test = splitter.split(X, y)
 
        # ----------------------------
        # Step 5 : Build Model
        # ----------------------------
 
        builder = ModelBuilder(
            model_type="lstm",
            input_shape=(12,1)
        )
 
        model = builder.build_model()
 
        # ----------------------------
        # Step 6 : Train Model
        # ----------------------------
 
        print("\nTraining Started...\n")
 
        try:
            history = model.fit(
                X_train,
                y_train,
                epochs=100,
                batch_size=8,
                validation_data=(X_test, y_test),
                verbose=1
            )
        except Exception as e:
            raise RuntimeError(f"Model training failed: {e}")
 
        print("\nTraining Completed Successfully.")
 
        # ----------------------------
        # Step 7 : Save Model
        # ----------------------------
 
        MODELS_DIR.mkdir(exist_ok=True)
        model.save(str(MODELS_DIR / "lstm_model.keras"))
        model.save(str(MODELS_DIR / "lstm_model.h5"))
 
        print("\nModel Saved Successfully.")
 
        return model, history
   
if __name__ == "__main__":
 
    trainer = ModelTrainer()
 
    model, history = trainer.train()