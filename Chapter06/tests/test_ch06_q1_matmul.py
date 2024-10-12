from turtledemo.forest import start

import numpy as np
import torch
import ctypes
import os
import time

from torch.xpu import device

current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_lib = ctypes.CDLL(os.path.join(current_dir, "libch06_q1_matmul.so"))

cuda_lib.matmul.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

def test_matmul():
    N = 10240
    M = 20480
    P = 10240

    A = np.random.rand(N, M).astype(np.float32)
    B = np.random.rand(M, P).astype(np.float32)
    C = np.zeros((N, P), dtype=np.float32)

    # Measure CUDA implementation time
    cuda_lib.matmul(A, B, C, N, M, P)
    start_time = time.time()
    cuda_lib.matmul(A, B, C, N, M, P)
    cuda_time = time.time() - start_time
    print(f"CUDA matmul time: {cuda_time:.4f} seconds")

    C_numpy = A @ B
    # Measure NumPy implementation time
    start_time = time.time()
    C_numpy = A @ B
    numpy_time = time.time() - start_time
    print(f"NumPy matmul time: {numpy_time:.4f} seconds")

    A_torch = torch.from_numpy(A).cuda()
    B_torch = torch.from_numpy(B).cuda()

    C_torch = A_torch@B_torch
    start_time = time.time()
    C_torch = A_torch@B_torch
    torch_time = time.time() - start_time
    print(f"PyTorch matmul time: {torch_time:.4f} seconds")


    # Validate results
    np.testing.assert_allclose(C_numpy, C, rtol=1e-4)

if __name__ == "__main__":
    test_matmul()
