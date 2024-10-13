## Checklist
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

## Q1
> Calculate the P[0] value in Fig. 7.3. 

P[0] = 8*5+2*3+5*1 = 46

## Q2
> Consider performing a 1D convolution on array N = {4,1,3,2,3} with filter F = {2,1,4}. 
> What is the resulting output array?

考虑padding：{8, 21, 13, 20, 7}

## Q3
>  What do you think the following 1D convolution filters are doing? 
> 
> a. [0 1 0] 
> 
> b. [0 0 1] 
> 
> c. [1 0 0] 
> 
> d. [ — 1/2 0 1/2] 
> 
> e. [1/3 1/3 1/3]

a. 不变

b. 向左移动一位

c. 向右移动一位

d. 计算梯度

e. 求均值

## Q4 
>Consider performing a 1D convolution on an array of size N with a filter of size M: 
> 
> a. How many ghost cells are there in total? 
> 
> b. How many multiplications are performed if ghost cells are treated as multiplications (by 0)? 
> 
> c. How many multiplications are performed if ghost cells are not treated as multiplications?

a. 默认M为奇数。 ghost cells 数量为 M-1 + M-3 + ... + 0 = (M-1)/2 * (M+1)/2 = (M^2-1)/4

b. N*M

c. M*M - (M^2-1)/4

## Q5
>Consider performing a 2D convolution on a square matrix of size N x N with a square filter of size M x M: 
> 
> a. How many ghost cells are there in total? 
> 
> b. How many multiplications are performed if ghost cells are treated as multiplications (by 0)? 
>
> c. How many multiplications are performed if ghost cells are not treated as multiplications?

a. ghost cells 数量不会算...

b. N*N*M*M

c. 不会

## Q6
> Consider performing a 2D convolution on a rectangular matrix of size N1 x N2 with a rectangular mask of size M1 x M2: 
> 
> a. How many ghost cells are there in total? 
> 
> b. How many multiplications are performed if ghost cells are treated as multiplications (by 0)? 
> 
> c. How many multiplications are performed if ghost cells are not treated as multiplications?

a. ghost cells 数量不会算...

b. N1*N2*M1*M2

c. 不会

## Q7
> Consider performing a 2D tiled convolution with the kernel shown in Fig. 7.12 on an array of size N x N with a filter
> of size M x M using an output tile of size T x T. 
> 
> a. How many thread blocks are needed? 
> 
> b. How many threads are needed per block? 
> 
> c. How much shared memory is needed per block? 
> 
> d. Repeat the same questions if you were using the kernel in Fig. 7.15.

a. ceil(N/T) * ceil(N/T)

b. (M+T-1)*(M+T-1)

c. (M+T-1)*(M+T-1)*sizeof(float)

d. ceil(N/T) * ceil(N/T); T*T; T*T*sizeof(float)

## Q8
> Revise the 2D kernel in Fig. 7.7 to perform 3D convolution.

见[ch07_q8_3dconv.cu](ch07_q8_3dconv.cu)

## Q9
> Revise the 2D kernel in Fig. 7.9 to perform 3D convolution.

见[ch07_q9_3dconv.cu](ch07_q9_3dconv.cu)

## Q10
> Revise the 2D kernel in Fig. 7.12 to perform 3D convolution.

见[ch07_q10_3dconv.cu](ch07_q10_3dconv.cu)


