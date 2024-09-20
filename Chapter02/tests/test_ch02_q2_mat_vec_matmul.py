import numpy as np
import ctypes
import os

cnt = 10
width = 4096

current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_lib = ctypes.CDLL(os.path.join(current_dir, "libch02_q2_mat_vec_matmul.so"))

cuda_lib.mat_vec_matmul.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_uint
]

def test_mat_vec_matmul():
    mat = np.random.rand(width, width).astype(np.float32)
    vec = np.random.rand(width).astype(np.float32)
    res = np.random.rand(width).astype(np.float32)

    for i in range(cnt):
        cuda_lib.mat_vec_matmul(mat, vec, res, width)
    np.testing.assert_allclose(res, mat @ vec, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_mat_vec_matmul()
    print("Pass")
