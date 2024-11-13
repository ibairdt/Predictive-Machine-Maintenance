import pandas as pd
import numpy as np
import pickle
from pathlib import Path

script_dir = Path(__file__).parent

# incoming dataset import
csv_path = script_dir.parent / 'data' / 'incoming' / 'incoming_data.csv'
try:
    df_incoming = pd.read_csv(csv_path, sep=";")
    print("CSV loaded")

except FileNotFoundError:
    print(f"CSV {csv_path} does not exist")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# model import
model_path = script_dir.parent / 'models' / 'rf_class.pkl'
try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
        print("Model loaded")

except FileNotFoundError:
    print(f"Model {model_path} does not exist")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# scaler import
scaler_path = script_dir.parent / 'scalers' / 'std_scaler.pkl'
try:
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
        print("Scaler loaded")

except FileNotFoundError:
    print(f"Scaler {scaler_path} does not exist")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# data preprocessing
df_incoming = df_incoming.drop(columns=["UDI", "Product ID", "Failure Type", "Air temperature [K]", "Rotational speed [rpm]"])

df_incoming = pd.get_dummies(data=df_incoming, columns=["Type"], drop_first=False, dtype=np.uint8)

df_incoming[["Process temperature [K]", "Torque [Nm]", "Tool wear [min]"]] = scaler.transform(df_incoming[["Process temperature [K]", "Torque [Nm]", "Tool wear [min]"]])

print(df_incoming)

# predictions
predictions = model.predict(df_incoming)
df_incoming["predictions"] = predictions

predictions_path = script_dir.parent / 'predictions' / 'predictions.csv'
df_incoming.to_csv(predictions_path, index=False)