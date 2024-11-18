import pandas as pd
import numpy as np
import pickle
from pathlib import Path

script_dir = Path(__file__).parent

# INCOMING DATASET IMPORT
csv_path = script_dir.parent / 'data' / 'incoming' / 'incoming_data.csv'
try:
    df_incoming = pd.read_csv(csv_path, sep=";")
    print("CSV loaded")

except FileNotFoundError:
    print(f"CSV {csv_path} does not exist")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# MODEL IMPORT
model_path = script_dir.parent / 'models' / 'rf_class.pkl'
try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
        print("Model loaded")

except FileNotFoundError:
    print(f"Model {model_path} does not exist")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# SCALER IMPORT
scaler_path = script_dir.parent / 'scalers' / 'std_scaler.pkl'
try:
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
        print("Scaler loaded")

except FileNotFoundError:
    print(f"Scaler {scaler_path} does not exist")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# DATA PREPROCESSING

# Drop columns we do not need for our prediction
df_incoming = df_incoming.drop(columns=["UDI", "Product ID", "Failure Type", "Air temperature [K]", "Rotational speed [rpm]"])

# One hot encoding of our Type variable
df_incoming = pd.get_dummies(data=df_incoming, columns=["Type"], drop_first=False, dtype=np.uint8)

# Apply scaler to three variables
df_incoming[["Process temperature [K]", "Torque [Nm]", "Tool wear [min]"]] = scaler.transform(df_incoming[["Process temperature [K]", "Torque [Nm]", "Tool wear [min]"]])


# PREDICTIONS

# Apply predict method of our model
predictions = model.predict(df_incoming)

# Add predictions column to the dataset including the predictions
df_incoming["predictions"] = predictions

# Export dataset with predictions as CSV file
predictions_path = script_dir.parent / 'predictions' / 'predictions.csv'
df_incoming.to_csv(predictions_path, index=False)