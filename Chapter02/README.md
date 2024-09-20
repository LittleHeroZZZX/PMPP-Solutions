## CheckList
- [x] Q1
- [x] Q2
- [ ] Q3
- [ ] Q4
- [ ] Q5

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
