import pandas as pd
import numpy as np
import pickle
from pathlib import Path

script_dir = Path(__file__).parent

# incoming dataset import
csv_path = script_dir.parent / 'data' / 'incoming' / 'incoming_data.csv'
if csv_path.exists():
    df_incoming = pd.read_csv(csv_path, sep=";")
    print("CSV loaded")
else:
    print(f"CSV {csv_path} does not exist")

# model import
model_path = script_dir.parent / 'models' / 'rf_class.pkl'
if model_path.exists():
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    print("Model loaded")
else:
    print(f"Model {model_path} does not exist")

# scaler import
scaler_path = script_dir.parent / 'scalers' / 'std_scaler.pkl'
if scaler_path.exists():
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Scaler loaded")
else:
    print(f"Scaler {scaler_path} does not exist")


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
