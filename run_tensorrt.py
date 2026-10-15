import json
import struct
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).parent
ENGINE = ROOT / "model" / "har_cnn_lstm.engine"
DATA = ROOT / "data" / "uci_test_windows.json"
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

windows = json.loads(DATA.read_text())["windows"]
window = windows[0]
values = [value for row in window["data"] for value in row]

with tempfile.TemporaryDirectory() as folder:
    input_file = Path(folder) / "input.raw"
    output_file = Path(folder) / "output.json"
    input_file.write_bytes(struct.pack(f"<{len(values)}f", *values))

    result = subprocess.run(
        [
            "trtexec",
            f"--loadEngine={ENGINE}",
            f"--loadInputs=sensor_window:{input_file}",
            f"--exportOutput={output_file}",
            "--duration=1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(output_file.read_text())[0]["values"]

predicted = max(range(len(output)), key=output.__getitem__)
latency_line = next(line for line in result.stdout.splitlines() if "Latency:" in line)
latency = latency_line.split("mean = ")[1].split(" ms")[0]

print(f"Expected: {window['label']}")
print(f"Predicted: {LABELS[predicted]} ({output[predicted]:.4f})")
print(f"TensorRT mean latency: {latency} ms")
