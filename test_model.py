from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "cnn_lstm.keras"
DATA_PATH = ROOT / "data" / "uci_test_windows.json"
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]
DEVICE = "/CPU:0"  # Change to "/CPU:0" to use CPU

if DEVICE == "/GPU:0" and not tf.config.list_physical_devices("GPU"):
    raise RuntimeError("TensorFlow cannot see a GPU")

def main() -> None:
    with DATA_PATH.open() as file:
        test_set = json.load(file)
    print(test_set.keys())
    windows = test_set["windows"]
    inputs = np.asarray([window["data"] for window in windows], dtype=np.float32)
    expected = np.asarray([LABELS.index(window["label"]) for window in windows])
    
    with tf.device(DEVICE):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        if tuple(inputs.shape[1:]) != tuple(model.input_shape[1:]):
            raise ValueError(f"Input shape {inputs.shape[1:]} does not match model {model.input_shape[1:]}")
        start = time.perf_counter()
        probabilities = model.predict(inputs, verbose=0)
        end = time.perf_counter()
        total_time = end - start
        print("total_time = ", total_time)
        predicted = probabilities.argmax(axis=1)
    # print(predicted)
        accuracy = float(np.mean(predicted == expected))

        print(f"Test windows: {len(windows)}")
        print(f"Overall accuracy: {accuracy:.3f}")
        for label_index, label in enumerate(LABELS):
            class_mask = expected == label_index
            class_accuracy = float(np.mean(predicted[class_mask] == expected[class_mask]))
            print(f"{label}: {class_accuracy:.3f} ({int(class_mask.sum())} windows)")


if __name__ == "__main__":
    main()
