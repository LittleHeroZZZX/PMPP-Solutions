## Checklist
- [x] Q1
- [x] Q2

## Q1
> Consider a 3D stencil computation on a grid of size 120 x 120 x 120, including boundary cells.
> 
> a. What is the number of output grid points that is computed during each stencil sweep?
> 
> b. For the basic kernel in Fig. 8.6, what is the number of thread blocks that are needed, 
> assuming a block size of 8 x 8 x 8?
> 
> c. For the kernel with shared memory tiling in Fig. 8.8, what is the number of thread 
> blocks that are needed, assuming a block size of 8 x 8 x 8?
> 
> d. For the kernel with shared memory tiling and thread coarsening in Fig. 8.10, what is the number
> of thread blocks that are needed, assuming a block size of 32 x 32?

a. 119 x 119 x 119 = 1,685,159

b. 120 / 8 = 15, 15^3 = 3375

c. 120 / 8 = 15, 15^3 = 3375

d. 120 / 32 = 4, 4\*4 = 16

## Q2
> Consider an implementation of a seven-point (3D) stencil with shared memory tiling and thread 
> coarsening applied. The implementation is similar to those in Figs. 8.10 and 8.12, except that 
> the tiles are not perfect cubes. Instead, a thread block size of 32 x 32 is used as well as a
> coarsening factor of 16 (i.e., each thread block processes 16 consecutive output planes in the 
> z dimension).
> 
> a. What is the size of the input tile (in number of elements) that the thread block loads 
> throughout its lifetime?
> 
> b. What is the size of the output tile (in number of elements) that the thread block 
> processes throughout its lifetime?
> 
> c. What is the floating point to global memory access ratio (in OP/B) of the kernel?
> 
> d. How much shared memory (in bytes) is needed by each thread block if register tiling is not used,
> as in Fig. 8.10?
> 
> e. How much shared memory (in bytes) is needed by each thread block if register tiling is used, 
> as in Fig. 8.12?

a. 32 x 32 x 18 = 18432

b. 30 x 30 x 16 = 14400

c. OPs: 13 x 14400; Bytes: 32 x 32 x 16 x 4; Ratio: 13 x 14400 / 32 x 32 x 16 x 4 = 2.86 OP/B

d. 32 x 32 x 3 x 4 = 12288 Bytes

e. 32 x 32 x 4 = 4096 Bytes