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

## Q1
Consider the following CUDA kernel and the corresponding host function that calls it:

```c
__global__ void foo_kernel(int* a, int* b) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(threadIdx.x < 40 || threadIdx.x >= 104) {
        b[i] = a[i] + 1;
    }
    if(i%2 == 0) {
        a[i] = b[i]*2;
    }
    for(unsigned int j = 0; j < 5 - (i%3); ++j) {
        b[i] += j;
    }
}

void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d);
}
```
a. What is the number of warps per block?

128/32 = 4

b. What is the number of warps in the grid?

(1024 + 128 - 1)/128 = 8 blocks

8 * 4 = 32 warps

c. For the statement in line 04:

i. How many warps in the grid are active?

In each block, only the 3rd block, with threadIdx.x ranges from 64 to 95, is fully inactive.

So 3*8 = 24 warps are active.

ii. How many warps in the SM are divergent?

Both the 2nd and the 4th wraps are divergent.

iii. What is the SIMD efficiency (in %) of warp 0 of block 0?

100%

iv. What is the SIMD efficiency (in %) of warp 1 of block 0?

(40-32)/32 = 25%

v. What is the SIMD efficiency (in %) of warp 3 of block 0?

0%

d. For the statement in line 07:

i. How many warps in the grid are active?
All 32 warps are active.

ii. How many warps in the SM are divergent?

All 32 warps are divergent.

iii. What is the SIMD efficiency (in %) of warp 0 of block 0?

50%

e. For the statement in line 10:

i. How many warps in the grid are active?

All 32 warps are active.

ii. How many warps in the SM are divergent?

None of the warps are divergent.

iii. What is the SIMD efficiency (in %) of warp 0 of block 0?

100%

## Q2
For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?

(2000 + 512 - 1)/512 * 512 = 2048

## Q3
For the previous question, how many warps do you expect to have divergence due to the boundary check on vector length?

Only the second to last block will diverge due to the boundary check.

## Q4
Consider a hypothetical block with 8 threads executing a section of code  before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and 2.9; they spend the rest of their time waiting for the barrier. What percentage of the threads' total execution time is spent waiting for the barrier?

(1+0.7+0+0.2+0.6+1.1+0.4+0.1)/24 = 17.1%

## Q5
A CUDA programmer says that if they launch a kernel with only 32 threads  in each block, they can leave out the __syncthreads() instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain.

Absolutely not. Execution time may vary even though the threads in a warp share the same code / instruction. E.g. the count of for-loop could be relevant to the thread index, which may lead to different execution time for each thread.

## Q6
If a CUDA device’s SM can take up to 1536 threads and up to 4 thread blocks, which of the following block configurations would result in the most number of threads in the SM? 

a. 128 threads per block 

b. 256 threads per block 

c. 512 threads per block 

d. 1024 threads per block

A: 128*4 = 512 threads, B: 256*4 = 1024 threads, C: 512*3 = 1536 threads, D: 1024*1 = 1024 threads

So the answer is C.

## Q7
Assume a device that allows up to 64 blocks per SM and 2048 threads per SM. Indicate which of the following assignments per SM are possible. In the cases in which it is possible, indicate the occupancy level. 

a. 8 blocks with 128 threads each 

b. 16 blocks with 64 threads each 

c. 32 blocks with 32 threads each 

d. 64 blocks with 32 threads each 

e. 32 blocks with 64 threads each

A: 8*128 = 1024 threads, 1024/2048 = 50% occupancy

B: 16*64 = 1024 threads, 1024/2048 = 50% occupancy

C: 32*32 = 1024 threads, 1024/2048 = 50% occupancy

D: 64*32 = 2048 threads, 2048/2048 = 100% occupancy

E: 32*64 = 2048 threads, 2048/2048 = 100% occupancy

So all of them are possible.

## Q8
Consider a GPU with the following hardware limits: 2048 threads per SM, 32 blocks per SM, and 64K (65,536) registers per SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor. 

a. The kernel uses 128 threads per block and 30 registers per thread. 

b. The kernel uses 32 threads per block and 29 registers per thread. 

c. The kernel uses 256 threads per block and 34 registers per thread.

A: 2048/128 = 16 blocks, 2048*30 = 61440 registers, full occupancy

B: 2048/32 = 64 blocks > 32, limited by the number of blocks

C: 2048/256 = 8 blocks, 2048*34 = 69632 registers, limited by the number of registers

## Q9
A student mentions that they were able to multiply two 1024 × 1024 matrices using a matrix multiplication kernel with 32 × 32 thread blocks. The student is using a CUDA device that allows up to 512 threads per block and up to 8 blocks per SM. The student further mentions that each thread in a thread block calculates one element of the result matrix. What would be your reaction and why?

The student is wrong. The number of threads in a block is 32*32 = 1024, which is larger than the limit of 512 threads per block.
