#include <stdio.h>

#define IN_TILE 8
#define RADIUS 3
#define OUT_TILE (IN_TILE - 2 * RADIUS)
#define KERNEL_SIZE (2 * RADIUS + 1)


__constant__ float d_kernel[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];

__global__ void conv3d_kernel(float *in, float *out, int depth, int height, int width, int r){
    __shared__ float in_tile[IN_TILE][IN_TILE][IN_TILE];
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;

    int depth_in = bz * OUT_TILE + tz - RADIUS;
    int row_in = by * OUT_TILE + ty - RADIUS;
    int col_in = bx * OUT_TILE + tx - RADIUS;
    __syncthreads();
    if(depth_in >=0 && depth_in < depth &&
        row_in >= 0 && row_in < height &&
        col_in >= 0 && col_in < width) {
        in_tile[tz][ty][tx] = in[depth_in * height * width + row_in * width + col_in];
    }else{
        in_tile[tz][ty][tx] = 0;
    }
    __syncthreads();

    int depth_tile = tz - RADIUS;
    int row_tile = ty - RADIUS;
    int col_tile = tx - RADIUS;
    if(depth_in >=0 && depth_in < depth &&
       row_in >= 0 && row_in < height &&
       col_in >= 0 && col_in < width) {
        if(depth_tile >=0 && depth_tile < OUT_TILE &&
           row_tile >= 0 && row_tile < OUT_TILE &&
           col_tile >= 0 && col_tile < OUT_TILE) {
            float sum = 0;
            for (int i = 0; i < KERNEL_SIZE; i++) {
                for (int j = 0; j < KERNEL_SIZE; j++) {
                    for (int k = 0; k < KERNEL_SIZE; k++) {
                        sum += in_tile[depth_tile + i][row_tile + j][col_tile + k] * d_kernel[i][j][k];
                    }
                }
            }
            out[depth_in * height * width + row_in * width + col_in] = sum;
        }
    }
}

extern "C" void conv3d(float *in, float *out, float *kernel, int depth, int height, int width, int r){
    float *d_in, *d_out;

    cudaMalloc(&d_in, depth * height * width * sizeof(float));
    cudaMalloc(&d_out, depth * height * width * sizeof(float));
    cudaMemcpy(d_in, in, depth * height * width * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_kernel, kernel, KERNEL_SIZE*KERNEL_SIZE*KERNEL_SIZE * sizeof(float));

    dim3 block(IN_TILE, IN_TILE, IN_TILE);
    dim3 grid((width + OUT_TILE - 1) / OUT_TILE, (height + OUT_TILE - 1) / OUT_TILE, (depth + OUT_TILE - 1) / OUT_TILE);
    conv3d_kernel<<<grid, block>>>(d_in, d_out, depth, height, width, r);
    cudaMemcpy(out, d_out, depth * height * width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}