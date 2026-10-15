import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx


ROOT = Path(__file__).resolve().parent
KERAS_PATH = ROOT / "model" / "cnn_lstm.keras"
ONNX_PATH = ROOT / "model" / "har_cnn_lstm.onnx"
DATA_PATH = ROOT / "data" / "uci_test_windows.json"
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]


def main():
	windows = json.loads(DATA_PATH.read_text())["windows"]
	inputs = np.asarray([window["data"] for window in windows], dtype=np.float32)
	expected = np.asarray([LABELS.index(window["label"]) for window in windows])

	model = tf.keras.models.load_model(KERAS_PATH, compile=False)
	signature = (tf.TensorSpec((1, 128, 9), tf.float32, name="sensor_window"),)
	model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=signature, opset=17)
	onnx.checker.check_model(model_proto)
	onnx.save(model_proto, ONNX_PATH)

	session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
	input_name = session.get_inputs()[0].name
	outputs = []
	start = time.perf_counter()
	for window in inputs:
		outputs.append(session.run(None, {input_name: window[None, ...]})[0][0])
	elapsed = time.perf_counter() - start

	probabilities = np.asarray(outputs)
	predicted = probabilities.argmax(axis=1)
	keras_outputs = model.predict(inputs, verbose=0)
	print(f"ONNX file: {ONNX_PATH.stat().st_size} bytes")
	print(f"ONNX Runtime CPU: {elapsed * 1000:.2f} ms total, {elapsed * 1000 / len(inputs):.3f} ms/window")
	print(f"Accuracy: {np.mean(predicted == expected):.3f}")
	print(f"Max probability difference vs Keras: {np.max(np.abs(probabilities - keras_outputs)):.3g}")


if __name__ == "__main__":
	main()
