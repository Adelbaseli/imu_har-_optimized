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
TFLITE_PATH = ROOT / "model" / "har_cnn_lstm.tflite"
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
    # print(test_set.keys())
    windows = test_set["windows"]
    inputs = np.asarray([window["data"] for window in windows], dtype=np.float32)
    expected = np.asarray([LABELS.index(window["label"]) for window in windows])
    
    with tf.device(DEVICE):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        fixed_input = tf.keras.Input(batch_shape=(1, 128, 9), dtype=tf.float32)
        fixed_model = tf.keras.Model(fixed_input, model(fixed_input, training=False))
        converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        TFLITE_PATH.write_bytes(tflite_model)

        if tuple(inputs.shape[1:]) != tuple(model.input_shape[1:]):
            raise ValueError(f"Input shape {inputs.shape[1:]} does not match model {model.input_shape[1:]}")
        start = time.perf_counter()
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        probabilities = []
        for window in inputs:
            interpreter.set_tensor(input_index, window[np.newaxis, ...])
            interpreter.invoke()
            probabilities.append(interpreter.get_tensor(output_index)[0])
        probabilities = np.asarray(probabilities)
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
