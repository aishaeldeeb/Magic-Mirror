import onnxruntime as ort

try:
    print("ONNX Runtime Version:", ort.__version__)
    print("Available Providers:", ort.get_available_providers())
except Exception as e:
    print(f"Error: {e}")
