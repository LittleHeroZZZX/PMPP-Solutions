#define BLOCK_SIZE 32

__global__ void mat_vec_matmul_kernel(float *mat, float* vec, float *out, int n){
    /*
     * calculate the matrix-vector multiplication of a matrix and a vector
     * each thread calculates the dot product of a row of the matrix and the vector
     * block and grid dimensions are set to 1D
     */
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n){
        float sum = 0;
        for (int i = 0; i < n; i++){
            sum += mat[row * n + i] * vec[i];
        }
        out[row] = sum;
    }
}

extern "C" void mat_vec_matmul(float *mat, float *vec, float *out, int n){
    /*
     * wrapper function for the matrix-vector multiplication kernel
     */
    int size = n * n * sizeof(float);
    float *d_mat, *d_vec, *d_out;
    cudaMalloc((void**)&d_mat, size);
    cudaMalloc((void**)&d_vec, n * sizeof(float));
    cudaMalloc((void**)&d_out, n * sizeof(float));
    cudaMemcpy(d_mat, mat, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vec, n * sizeof(float), cudaMemcpyHostToDevice);
    mat_vec_matmul_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_mat, d_vec, d_out, n);
    cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_out);
}