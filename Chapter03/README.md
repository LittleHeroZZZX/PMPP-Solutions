## CheckList
- [x] Q1
- [x] Q2
- [x] Q3
- [x] Q4
- [x] Q5

## Q1
In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them. 

a. Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design. 

b. Write a kernel that has each thread produce one output matrix column. Fill in the execution configuration parameters for the design. 

c. Analyze the pros and cons of each of the two kernel designs.

1. row-per-thread kernel
Pros:
- Memory coalescing: Each thread accesses contiguous memory locations in M

Cons:
- Low parallelism for matrices with few rows but many columns
- Poor memory coalescing for N

2. column-per-thread kernel
Pros:
- Memory coalescing: Each thread accesses contiguous memory locations in M
Cons:
- Low parallelism for matrices with few columns but many rows
- Poor memory coalescing for N

3. benchmark

   | Matrix size | row-per-thread (s) | column-per-thread (s) | np.matmul (s) |
   |-------------|--------------------|-----------------------|---------------|
   | 32          | 473e-6             | 482e-6                | 2.9e-6        |
   | 256         | 24e-4              | 15e-4                 | 2.2e-4        |
   | 1024        | 33e-3              | 21e-3                 | 4.3e-3        |
   | 4096        | 13                 | 7                     | 1.6           |

I fail to explain that row-per-thread behaves a little bit better than column-per-thread, and the gap increases as the matrix size increases.

Unsurprisingly, np.matmul is about 5x faster than my implementations.

## Q2
A matrix-vector multiplication takes an input matrix B and a vector C and produces one output vector A. Each element of the output vector A is the dot  product of one row of the input matrix B and C, that is, $A[i] = \sum^j B[i][j] \cdot C[j]$. For simplicity we will handle only square matrices whose elements are singleprecision floating-point numbers. Write a matrix-vector multiplication kernel and the host stub function that can be called with four parameters: pointer to the output matrix, pointer to the input matrix, pointer to the input vector, and the number of elements in each dimension. Use one thread to calculate an output vector element.

## Q3
Consider the following CUDA kernel and the corresponding host function that calls it:
```c
__global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        b[row * N + col] = a[row * N + col] / 2.1f + 4.8f;
    }
}

void foo(float* a_d, float* b_d) {
    unsigned int M = 150;
    unsigned int N = 300;
    dim3 bd(16, 32);
    dim3 gd((N - 1) / 16 + 1, (M - 1) / 32 + 1);
    foo_kernel<<<gd, bd>>>(a_d, b_d, M, N);
}
```
a. What is the number of threads per block? 

16*32 = 512

b. What is the number of threads in the grid

(300-1)/16 + 1 = 19
(150-1)/32 + 1 = 5
19*5*512 = 48640

c. What is the number of blocks in the grid?  

19*5 = 95

d. What is the number of threads that execute the code on line 05?

300*150 = 45000

## Q4
Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of the matrix element at row 20 and column 10: 

a. If the matrix is stored in row-major order. 

20*400 + 10 = 8010

b. If the matrix is stored in column-major order.

10*500 + 20 = 5020

## Q5
Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at x = 10, y = 20, and = 5.

10*500*300 + 20*300 + 5 = 1505005
