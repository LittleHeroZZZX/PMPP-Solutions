#define BLOCK_SIZE 32

__global__ void matmul_row_kernel(float *M, float *N, float *P, int width){
    /*
     * each thread compute one row of P
     * both block and grid size are 1D
     * i-th row of P involves the i-th row of M and all columns of N
     */
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if(row < width){
        for(int j=0; j<width; j++){ // j iterates over N's columns
            float sum = 0;
            for(int i=0; i<width; i++){ // i iterates over P's columns / M's columns
                sum += M[row*width+i] * N[i*width+j];
            }
            P[row*width+j] = sum;
        }
    }
}

__global__ void matmul_column_kernel(float *M, float *N, float *P, int width){
    /*
     * each thread compute one column of P
     * both block and grid size are 1D
     * i-th column of P involves the i-th column of N and all rows of M
     */
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if(col < width){
        for(int j=0; j<width; j++){ // j iterates over M's rows
            float sum = 0;
            for(int i=0; i<width; i++){ // i iterates over P's rows / N's rows
                sum += M[j*width+i] * N[i*width+col];
            }
            P[j*width+col] = sum;
        }
    }
}

extern "C" void matmul_row(float *M, float *N, float *P, int width){
    float *d_M, *d_N, *d_P;
    int size = width * width * sizeof(float);

    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);

    cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);

    matmul_row_kernel<<<(width+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_M, d_N, d_P, width);

    cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

extern "C" void matmul_column(float *M, float *N, float *P, int width){
    float *d_M, *d_N, *d_P;
    int size = width * width * sizeof(float);

    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);

    cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);

    matmul_column_kernel<<<(width+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_M, d_N, d_P, width);

    cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}