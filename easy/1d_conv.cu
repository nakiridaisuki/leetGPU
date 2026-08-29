#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#define IDX_X (blockIdx.x * blockDim.x + threadIdx.x)
#define IDX_Y (blockIdx.y * blockDim.y + threadIdx.y)

#define INPUT_SIZE 1500000
#define KERNEL_SIZE 2048
#define BLOCK_DIM 256
#define MIN(a, b) (a < b ? a : b)

__constant__ float const_kernel[KERNEL_SIZE];

__global__ void convolution_1d_kernel(
    const float *input,
    const float *kernel,
    float *output,
    int input_size,
    int kernel_size
) {
    int idx = IDX_X;

    __shared__ float block[BLOCK_DIM + KERNEL_SIZE - 1];
    int input_total =
        MIN(BLOCK_DIM + KERNEL_SIZE - 1,
            input_size - (blockIdx.x * blockDim.x));
    int block_start_idx = blockIdx.x * blockDim.x;
    for (int i = threadIdx.x; i < input_total; i += blockDim.x) {
        block[i] = input[block_start_idx + i];
    }
    __syncthreads();

    if (idx < input_size - kernel_size + 1) {
        float tmp = 0;
        for (int i = 0; i < kernel_size; i++) {
            tmp += block[threadIdx.x + i] * const_kernel[i];
        }
        output[idx] = tmp;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the
// GPU)
void solve(
    const float *input,
    const float *kernel,
    float *output,
    int input_size,
    int kernel_size
) {
    int output_size     = input_size - kernel_size + 1;
    int threadsPerBlock = BLOCK_DIM;
    int blocksPerGrid   = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, kernel, output, input_size, kernel_size
    );
    cudaDeviceSynchronize();
}

void print_mat(const float *mat, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            printf("%.2f, ", mat[i * m + j]);
        printf("\n");
    }
}

int main() {
    const int input_len = INPUT_SIZE, kernel_len = KERNEL_SIZE;
    const int output_len = (input_len - kernel_len + 1);
    srand(1000);

    float *input, *kernel, *output;
    cudaMallocManaged(&input, input_len * sizeof(float));
    cudaMallocManaged(&kernel, kernel_len * sizeof(float));
    cudaMallocManaged(&output, output_len * sizeof(float));

    for (int i = 0; i < input_len; i++) {
        input[i] = rand() % 10;
    }
    for (int i = 0; i < kernel_len; i++) {
        kernel[i] = rand() % 10 - 5;
    }

    cudaMemcpyToSymbol(const_kernel, kernel, kernel_len * sizeof(float));

    solve(input, kernel, output, input_len, kernel_len);

    // printf("input = {\n");
    // print_mat((float *)input, 1, input_len);
    // printf("}\n");
    // printf("kernel = {\n");
    // print_mat((float *)kernel, 1, kernel_len);
    // printf("}\n");
    // printf("output = {\n");
    // print_mat((float *)output, 1, output_len);
    // printf("}\n");

    for (int i = 0; i < output_len; i++) {
        float tmp = 0;
        for (int j = 0; j < kernel_len; j++) {
            tmp += input[i + j] * kernel[j];
        }
        if (output[i] != tmp) {
            printf(
                "ERROR answer at index %d has %.2f should be %.2f\n",
                i,
                output[i],
                tmp
            );
            return 1;
        }
    }

    cudaFree(input);
    cudaFree(kernel);
    cudaFree(output);
}
