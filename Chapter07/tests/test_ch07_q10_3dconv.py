import numpy as np
import torch
import ctypes
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_lib = ctypes.CDLL(os.path.join(current_dir, "libch07_q10_3dconv.so"))

cuda_lib.conv3d.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

def test_conv_3d():
    DEPTH = 1024
    HEIGHT = 1024
    WIDTH = 64
    RADIUS = 3

    input = np.random.rand(DEPTH, HEIGHT, WIDTH).astype(np.float32)
    kernel = np.random.rand(2*RADIUS+1, 2*RADIUS+1, 2*RADIUS+1).astype(np.float32)
    output = np.zeros((DEPTH, HEIGHT, WIDTH), dtype=np.float32)

    cuda_lib.conv3d(input, output, kernel, DEPTH, HEIGHT, WIDTH, RADIUS)

    # Calculate true_out using PyTorch
    input_tensor = torch.tensor(input).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    kernel_tensor = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Perform 3D convolution
    true_out_tensor = torch.nn.functional.conv3d(input_tensor, kernel_tensor, padding=RADIUS)
    true_out = true_out_tensor.squeeze().numpy()  # Remove extra dimensions

    np.testing.assert_allclose(true_out, output, rtol=1e-4)

if __name__ == "__main__":
    test_conv_3d()
