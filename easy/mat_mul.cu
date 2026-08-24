#include <cuda_runtime.h>
#include <iostream>

#define IDX_X (blockIdx.x * blockDim.x + threadIdx.x)
#define IDX_Y (blockIdx.y * blockDim.y + threadIdx.y)

__global__ void matrix_multiplication_kernel(
    const float *A, const float *B, float *C, int M, int N, int K
) {
    int i = IDX_Y, j = IDX_X;
    if (i < M && j < K) {
        float sum = 0;
        for (size_t t = 0; t < N; t++)
            sum += A[i * N + t] * B[t * K + j];
        C[i * K + j] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (K + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A, B, C, M, N, K
    );
    cudaDeviceSynchronize();
}

int main() {
    int n = 2, m = 2;
    size_t size = n * m * sizeof(float);

    float h_A[2][2] = {{1, 2}, {3, 4}};
    float h_B[2][2] = {{5, 6}, {7, 8}};
    float h_C[2][2];
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    solve(d_A, d_B, d_C, 2, 2, 2);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result of C = {\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            printf("%.2f, ", h_C[i][j]);
        printf("\n");
    }
    printf("}\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
