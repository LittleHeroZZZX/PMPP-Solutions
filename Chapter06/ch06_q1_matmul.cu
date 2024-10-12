#define TILE_WIDTH 32

__global__ void matmul_kernel(float* A, float *B, float *C, int m, int n, int p){
    /*
     * calculate C = A@B, where A is m x n, B is n x p, and C is m x p
     * use coalescing when loading input tile from global mem to shared mem
     */
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float p_val = 0;
    for (int i=0; i<(n+TILE_WIDTH-1)/TILE_WIDTH; i++){
        __syncthreads();
        // load tile from A row by row
        if (row < m && i*TILE_WIDTH+tx < n){
            As[ty][tx] = A[row*n + i*TILE_WIDTH + tx];
        } else {
            As[ty][tx] = 0;
        }

        // load tile from B row by row
        if (i*TILE_WIDTH < n && col<p){
            Bs[ty][tx] = B[(i*TILE_WIDTH + ty)*p + col];
        } else {
            Bs[ty][tx] = 0;
        }

        __syncthreads();

        for (int j=0; j<TILE_WIDTH; j++) {
            p_val += As[ty][j] * Bs[j][tx];
        }
    }
    C[row*p + col] = p_val;
}

extern "C" void matmul(float* A, float *B, float *C, int m, int n, int p){
    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, m*n*sizeof(float));
    cudaMalloc(&B_d, n*p*sizeof(float));
    cudaMalloc(&C_d, m*p*sizeof(float));

    cudaMemcpy(A_d, A, m*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, n*p*sizeof(float), cudaMemcpyHostToDevice);


    dim3 dimGrid((p+TILE_WIDTH-1)/TILE_WIDTH, (m+TILE_WIDTH-1)/TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    matmul_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, m, n, p);

    cudaMemcpy(C, C_d, m*p*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}