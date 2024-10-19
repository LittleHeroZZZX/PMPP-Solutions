## Checklist
- [x] Q1
- [x] Q2
- [x] Q3
- [x] Q4
- [x] Q5
- [x] Q6

## Q1
> Assume that each atomic operation in a DRAM system has a total latency of 100 ns.
> What is the maximum throughput that we can get for atomic operations on the same
> global memory variable?

单个原子操作包括一次读写，即单个原子操作的最小延迟为200ns，因此最大吞吐量为1/200ns = 5M ops/s

## Q2
> For a processor that supports atomic operations in L2 cache, assume that each atomic
> operation takes 4 ns to complete in L2 cache and 100 ns to complete in DRAM. 
> Assume that 90% of the atomic operations hit in L2 cache. What is the approximate 
> throughput for atomic operations on the same global memory variable?

90%的原子操作在L2缓存中命中，因此90%的原子操作延迟为4ns，10%的原子操作延迟为100ns，
因此平均延迟为0.9*4ns+0.1*100ns=13.6ns，因此吞吐量为1/13.6ns=73.5M ops/s

## Q3
> In Exercise 1, assume that a kernel performs five floating-point operations per atomic
> operation. What is the maximum floating-point throughput of the kernel execution as limited 
> by the throughput of the atomic operations?

5 x 5M ops/s = 25M ops/s

## Q4
> In Exercise 1, assume that we privatize the global memory variable into shared memory variables
> in the kernel and that the shared memory access latency is 1 ns. All original global memory atomic
> operations are converted into shared memory atomic operation. For simplicity, assume that the additional
> global memory atomic operations for accumulating privatized variable into the global variable adds
> 10% to the total execution time. Assume that a kernel performs five floating-point operations per
> atomic operation. What is the maximum floating-point throughput of the kernel execution as limited
> by the throughput of the atomic operations?

单个共享内存原子操作时延为2ns，平均时延为2/0.9 ns, 吞吐量为1/(2/0.9) = 450M ops/s。浮点吞吐量为5 x 450M ops/s = 2.25G ops/s

## Q5
> To perform an atomic add operation to add the value of an integer variable Partial to a global memory 
> integer variable Total, which one of the following statements should be used? 
> 
> a. atomicAdd(Total, 1); 
> 
> b. atomicAdd(&Total, &Partial); 
> 
> c. atomicAdd(Total, &Partial); 
> 
> d. atomicAdd(&Total, Partial);

d

## Q6
> Consider a histogram kernel that processes an input with 524,288 elements to produce a 
> histogram with 128 bins. The kernel is configured with 1024 threads per block. 
> 
> a. What is the total number of atomic operations that are performed on 
> global memory by the kernel in Fig. 9.6 where no privatization, shared memory, 
> and thread coarsening are used?
> 
> b. What is the maximum number of atomic operations that may be performed on global memory by the kernel in 
> Fig. 9.10 where privatization and shared memory are used but not thread coarsening? 
> 
> c. What is the maximum number of atomic operations that may be performed on global memory by the kernel in 
> Fig. 9.14 where privatization, shared memory, and thread coarsening are used with a coarsening factor of 4?

a. (524288 + 128 - 1) / 128 = 4096

b. 共有512个block，每个block进行128次原子操作，共计512 x 128 = 65536次原子操作

c. 共有128个block，每个block进行128次原子操作，共计128 x 128 = 16384次原子操作


