import ctypes
try:
    ctypes.CDLL(r"C:\Program Files\NVIDIA\CUDNN\v9.5\bin\cudnn64_9.dll")
    print("cuDNN loaded successfully.")
except OSError as e:
    print(f"Error loading cuDNN: {e}")

try:
    ctypes.CDLL(r"C:\Windows\System32\nvcuda.dll")
    print("CUDA loaded successfully.")
except OSError as e:
    print(f"Error loading CUDA: {e}")
