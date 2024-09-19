import numpy as np
import ctypes
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_lib = ctypes.CDLL(os.path.join(current_dir, "libch02_q1_matmul.so"))

cuda_lib.matmul_row.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_uint
    ]

cuda_lib.matmul_column.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_uint
    ]

def test_matmul_row():
    width = 1000
    M = np.random.rand(width, width).astype(np.float32)
    N = np.random.rand(width, width).astype(np.float32)
    P = np.random.rand(width, width).astype(np.float32)
    cuda_lib.matmul_row(M, N, P, width)

    np.testing.assert_allclose(np.matmul(M, N), P, rtol=1e-3)

def test_matmul_column():
    width = 1000
    M = np.random.rand(width, width).astype(np.float32)
    N = np.random.rand(width, width).astype(np.float32)
    P = np.random.rand(width, width).astype(np.float32)
    cuda_lib.matmul_column(M, N, P, width)

    np.testing.assert_allclose(np.matmul(M, N), P, rtol=1e-3)

if __name__ == "__main__":
    test_matmul_row()
    test_matmul_column()
    print("Pass")