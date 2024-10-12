## Checklist
- [x] Q1
- [x] Q2
- [x] Q3
- [x] Q4

## Q1
> Write a matrix multiplication kernel function that corresponds to the design illustrated in Fig. 6.4.

见 [ch06_q1_matmul.cu](./ch06_q1_matmul.cu).

## Q2
>For tiled matrix multiplication, of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE
>will the kernel completely avoid uncoalesced accesses to global memory? (You need to consider only square blocks.)

当`BLOCK_SIZE`是线程束的整数倍即32的整数倍时，单个线程束内的线程访问的地址必定连续，可以启用合并内存访问。

## Q3
> Consider the following CUDA kernel:
> ```c
> 01 __global__ void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
> 02 unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
> 03 __shared__ float a_s[256];
> 04 __shared__ float bc_s[4*256];
> 05 a_s[threadIdx.x] = a[i];
> 06 for(unsigned int j = 0; j < 4; ++j) {
> 07 bc_s[j*256 + threadIdx.x] = b[j*blockDim.x*gridDim.x + i] + c[i*4 + j];
> 08 }
> 09 __syncthreads();
> 10 d[i + 8] = a_s[threadIdx.x];
> 11 e[i*8] = bc_s[threadIdx.x*4];
> 12 }
> 
> ```
> For each of the following memory accesses, specify whether they are coalesced or uncoalesced or coalescing 
> is not applicable: 
>
> a. The access to array a of line 05 
> 
> b. The access to array a_s of line 05 
> 
> c. The access to array b of line 07 
> 
> d. The access to array c of line 07 
> 
> e. The access to array bc_s of line 07 
> 
> f. The access to array a_s of line 10 
> 
> g. The access to array d of line 10 
> 
> h. The access to array bc_s of line 11 
> 
> i. The access to array e of line 11

a. 是合并访问

b. `a_s`是共享内存，合并访问不适用

c. 是合并访问

d. 不是合并访问

e. `bc_s`是共享内存，合并访问不适用

f. `a_s`是共享内存，合并访问不适用

g. 是合并访问

h. `bc_s`是共享内存，合并访问不适用

i. 不是合并访问

## Q4
> What is the floating point to global memory access ratio (in OP/B) of each of the following matrix-matrix 
> multiplication kernels? 
> 
> a. The simple kernel described in Chapter 3, Multidimensional Grids and Data, without any optimizations applied. 
> 
> b. The kernel described in Chapter 5, Memory Architecture and Data Locality, with shared memory tiling applied using 
> a tile size of 32 x 32. 
> 
> c. The kernel described in this chapter with shared memory tiling applied using a tile size of 32 x 32 and 
> thread coarsening applied using a coarsening factor of 4.

a. 第八行每次计算进行两次浮点数读取，一次浮点乘法、一次加法，因此比值为2 OP/ 8 B = 0.25 OP/B

b. 每个block计算一个TILE时，加载了TILE\*TILE\*2个浮点数到共享内存中，block中每个线程计算一个元素需要进行TILE次浮点乘法和TILE浮点加法，
因此block中一共进行了TILE\*TILE\*2TILE次浮点操作。比值为2 TILE^3 OP / 2 TILE^2 * 4 B = TILE/4 OP/B 即 8 OP/B。

c. 以粗化因子=4为例进行粗化。经过粗化以后，每个block只要从A中加载一次TILE\*TILE个浮点数到共享内存中，
从B中加载`FACTOR`次TILE个浮点数到共享内存中。
进行了FACTOR\*TILE\*TILE\*2TILE浮点操作。比值为2 FACTOR \* TILE^3/ (FACTOR+1) TILE^2 * 4 B  =  FACTOR \* TILE/2 (FACTOR+1)
=12.8 OP/B
