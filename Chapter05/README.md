## CheckList
- [x] Q1
- [x] Q2
- [x] Q3
- [x] Q4
- [x] Q5
- [x] Q6
- [x] Q7
- [ ] Q8
- [ ] Q9
- [ ] Q10
- [ ] Q11
- [ ] Q12

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
