__global__ void matmul_row_kernel(float *M, float *N, float *P, int width){
    /*
     * each thread compute one row of P
     * both block and grid size are 1D
     * i-th row of P involves the i-th row of M and all columns of N
     */
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if(row < width){
        int row_offset = row * width;
        for(int j=0; j<width; j++){
            P[row_offset+j] = 0;
        }
        for(int i=0; i<width; i++){ // i iterates M's columns
            for(int j=0; j<width; j++) { // j iterates over P's columns / N's columns
                P[row_offset+j] += M[row_offset+i] * N[i*width+j];
            }
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

    matmul_row_kernel<<<width, width>>>(d_M, d_N, d_P, width);

    cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}