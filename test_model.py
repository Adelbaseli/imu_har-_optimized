import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "cnn_lstm.keras"
DATA_PATH = ROOT / "data" / "uci_test_windows.json"
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

with DATA_PATH.open() as file:
    windows = json.load(file)["windows"]

inputs = np.asarray([window["data"] for window in windows], dtype=np.float32)
expected = np.asarray([LABELS.index(window["label"]) for window in windows])
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
predicted = model.predict(inputs, verbose=0).argmax(axis=1)

print(f"Test windows: {len(windows)}")
print(f"Overall accuracy: {np.mean(predicted == expected):.3f}")
for index, label in enumerate(LABELS):
    mask = expected == index
    print(f"{label}: {np.mean(predicted[mask] == expected[mask]):.3f} ({mask.sum()} windows)")
