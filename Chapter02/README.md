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
