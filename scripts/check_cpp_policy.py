"""Compile the real C++ ONNX runner and compare recurrent actions with Python ORT."""

from pathlib import Path
import subprocess
import tempfile
import argparse

import numpy as np
import onnxruntime as ort


def check(model: str) -> float:
  root = Path(__file__).resolve().parents[1]
  runtime = root / "deploy/thirdparty/onnxruntime-linux-x64-1.22.0"
  import platform
  if platform.machine() == "aarch64":
    runtime = root / "deploy/thirdparty/onnxruntime-linux-aarch64-1.22.0"
  with tempfile.TemporaryDirectory(prefix="pgtt-cpp-") as temp:
    exe = str(Path(temp) / "check_policy")
    subprocess.run(["g++", "-std=c++17", "-O2", "-pthread", "-I", str(root / "deploy/include"),
                    "-I", str(runtime / "include"), str(root / "tests/cpp/test_policy_runtime.cpp"),
                    str(runtime / "lib/libonnxruntime.so.1.22.0"),
                    "-Wl,-rpath," + str(runtime / "lib"), "-o", exe], check=True)
    output = subprocess.check_output([exe, str(Path(model).resolve())], text=True)
  actual = np.array([[float(v) for v in line.split()[1:]] for line in output.splitlines()
                     if line.startswith("ACTION ")])
  session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
  hidden = np.zeros((1, 1, 256), dtype=np.float32)
  expected = []
  for step in range(10):
    if step == 5:
      hidden.fill(0)
    obs = np.sin((np.arange(153, dtype=np.float32) + step) * np.float32(0.1))[None]
    actions, hidden = session.run(None, {"obs": obs, "h_in": hidden})
    expected.append(actions[0])
  error = float(np.max(np.abs(actual - np.array(expected))))
  if not np.isfinite(error) or error > 1e-5:
    raise RuntimeError(f"C++/Python recurrent action mismatch {error}")
  print(f"C++/Python max action error: {error:.3g}; dimension/NaN/map fallback checks passed")
  return error


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("model")
  check(parser.parse_args().model)
