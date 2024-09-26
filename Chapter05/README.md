## CheckList
- [x] Q1
- [x] Q2
- [x] Q3
- [x] Q4
- [x] Q5
- [x] Q6
- [x] Q7
- [x] Q8
- [x] Q9
- [x] Q10
- [x] Q11
- [x] Q12

## Q1
Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption?
Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality 
between threads.

Can't. Threads residing in the same block do not share any duplicated data.

## Q2
Draw the equivalent of Fig. 5.7 for a 8 × 8 matrix multiplication with 2 × 2 tiling and 4 × 4 tiling. 
Verify that the reduction in global memory bandwidth is indeed proportional to the dimension size of 
the tiles.

Figure is omitted. In the original version, each thread calls for 8+8 element access, so all 64 threads
access global memory 16×64=1024 times. In the 2x2 tiling version, each thread fetch 2 elements into shared
memory per phase. There're 4 phases, so all 64 threads access global memory 4×2×64=512 times.
In the 4x4 tiling version, each thread fetch 4 elements into shared memory per phase. There're 4 phases, so
all 64 threads access global memory 4×4×64=256 times, 1/4 of the original version.

Obviously, the reduction in global memory bandwidth is indeed proportional to the dimension size of the tiles.

## Q3
What type of incorrect execution behavior can happen if one forgot to use one or both
`__syncthreads()` in the kernel of Fig. 5.9?

Those threads that already finished their work(load data or compute) will enter the next phase, where they may 
try to access unloaded shared data or write to the shared memory being used by other threads, thus causing
incorrect execution behavior.

## Q4
Assuming that capacity is not an issue for registers or shared memory, give one important reason why
it would be valuable to use shared memory instead of registers to hold values fetched from global memory?
Explain your answer.

First, shared memory can be leveraged as a way of communication between threads.

Second, fetching data concurrently form global memory puts high pressure on global memory bus and can be
limited by the bandwidth. Shared memory based co-fetching is able to avoid duplicated data loading.

## Q5
For our tiled matrix-matrix multiplication kernel, if we use a 32 × 32 tile, what is the reduction 
of memory bandwidth usage for input matrices M and N?

1/32 of the original version.

## Q6
Assume that a CUDA kernel is launched with 1000 thread blocks, each of which has 512 threads. 
If a variable is declared as a local variable in the kernel, how many versions of the variable
will be created through the lifetime of the execution of the kernel?

1000×512=512000 versions.

## Q7
In the previous question, if a variable is declared as a shared memory variable, how many 
versions of the variable will be created through the lifetime of the execution of the kernel?

1000 versions.

## Q8
Consider performing a matrix multiplication of two input matrices with dimensions N × N. 
How many times is each element in the input matrices requested from global memory when: 

a. There is no tiling? b. Tiles of size T × T are used?

a. N times

b. N/T times

## Q9
A kernel performs 36 floating-point operations and seven 32-bit global memory accesses per thread. 
For each of the following device properties, indicate whether this kernel is compute-bound or memorybound. 
a. Peak FLOPS=200 GFLOPS, peak memory bandwidth=100 GB/second 
b. Peak FLOPS=300 GFLOPS, peak memory bandwidth=250 GB/second

Kernel compute intensity: 36/28 = 1.28 OP/B

a. Max compute intensity: 200/100 = 2 OP/B, so it's memory-bound.

b. Max compute intensity: 300/250 = 1.2 OP/B, so it's compute-bound.

## Q10
To manipulate tiles, a new CUDA programmer has written a device kernel that will transpose each 
tile in a matrix. The tiles are of size BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of 
matrix A is known to be a multiple of BLOCK_WIDTH. The kernel invocation and code are shown below. 
BLOCK_WIDTH is known at compile time and could be set anywhere from 1 to 20.

```c
dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
dim3 gridDim(A_width/blockDim.x, A_height/blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

__global__ void
BlockTranspose(float* A_elements, int A_width, int A_height)
{
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];

    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
```
a. Out of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will this 
kernel function execute correctly on the device? 

b. If the code does not execute correctly for all BLOCK_SIZE values, what is the root cause 
of this incorrect execution behavior? Suggest a fix to the code to make it work for all BLOCK_SIZE values.

a. BLOCK_SIZE=1

b. The root cause is lack of synchronization between data fetching phase and result writing phase.

## Q11
Consider the following CUDA kernel and the corresponding host function that calls it:

```c
__global__ void foo_kernel(float* a, float* b) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    float x[4];
    __shared__ float y_s;
    __shared__ float b_s[128];
    for(unsigned int j = 0; j < 4; ++j) {
        x[j] = a[j*blockDim.x*gridDim.x + i];
    }
    if(threadIdx.x == 0) {
        y_s = 7.4f;
    }
    b_s[threadIdx.x] = b[i];
    __syncthreads();
    b[i] = 2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3]
         + y_s*b_s[threadIdx.x] + b_s[(threadIdx.x + 3)%128];
}

void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d);
}
```

a. How many versions of the variable i are there? 

b. How many versions of the array x[] are there? 

c. How many versions of the variable y_s are there? 

d. How many versions of the array b_s[] are there?

e. What is the amount of shared memory used per block (in bytes)?

f. What is the floating-point to global memory access ratio of the kernel (in OP/B)?

a. 1024

b. 1024

c. 8

d. 8

e. 4+128*4=516

f. 5 times of float add and 5 times of float mul, total 10 times of float OP. From top
to bottom, 4 times of a[] read, 1 time of b[] read, 1 time of b write, 
total 6 times of global memory access. So the ratio is 10/6/4=0.4167 OP/B.

## Q12
Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM, 
64K (65,536) registers/SM, and 96 KB of shared memory/SM. For each of the following 
kernel characteristics, specify whether the kernel can achieve full occupancy. 
If not, specify the limiting factor. 

a. The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared memory/SM.

The kernel demands 2048/64 = 32 blocks/SM, 27\*2k=54k registers/SM. So it can achieve full occupancy.

b. The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of shared memory/SM.

The kernel demands 2048/256 = 8 blocks/SM, 31\*2k=62k registers/SM, 
8k shared memory/SM. It can achieve full occupancy.

