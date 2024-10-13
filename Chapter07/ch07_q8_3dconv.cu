#define TILE 8

__global__ void conv3d_kernel(float *in, float *out, float *kernel, int depth, int height, int width, int r){
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_depth = blockIdx.z * blockDim.z + threadIdx.z;
    float sum = 0.0f;
    for (int i=-r; i<=r; i++){
        for (int j=-r; j<=r; j++){
            for (int k=-r; k<=r; k++){
                int in_depth = out_depth + i;
                int in_row = out_row + j;
                int in_col = out_col + k;
                if (in_depth >= 0 && in_depth < depth && in_row >= 0 && in_row < height && in_col >= 0 && in_col < width){
                    sum += in[in_depth * height * width + in_row * width + in_col] * kernel[(i+r) * (2*r+1) * (2*r+1) + (j+r) * (2*r+1) + (k+r)];
                }
            }
        }
    }
    if (out_row < height && out_col < width && out_depth < depth){
        out[out_depth * height * width + out_row * width + out_col] = sum;
    }
}

extern "C" void conv3d(float *in, float *out, float *kernel, int depth, int height, int width, int r){
    float *d_in, *d_out, *d_kernel;
    cudaMalloc(&d_in, depth * height * width * sizeof(float));
    cudaMalloc(&d_out, depth * height * width * sizeof(float));
    cudaMalloc(&d_kernel, (2*r+1) * (2*r+1) * (2*r+1) * sizeof(float));
    cudaMemcpy(d_in, in, depth * height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, (2*r+1) * (2*r+1) * (2*r+1) * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(TILE, TILE, TILE);
    dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE, (depth + TILE - 1) / TILE);
    conv3d_kernel<<<grid, block>>>(d_in, d_out, d_kernel, depth, height, width, r);
    cudaMemcpy(out, d_out, depth * height * width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);
}