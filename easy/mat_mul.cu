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

void print_mat(const float *mat, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            printf("%.2f, ", mat[i * m + j]);
        printf("\n");
    }
}

int main() {
    int n = 1024, m = 1024, k = 1024;
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(n * m * sizeof(float));
    h_B = (float *)malloc(m * k * sizeof(float));
    h_C = (float *)malloc(n * k * sizeof(float));
    for (int i = 0; i < n * m; i++)
        h_A[i] = rand() % 5;
    for (int i = 0; i < m * k; i++)
        h_B[i] = rand() % 5;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, n * m * sizeof(float));
    cudaMalloc((void **)&d_B, m * k * sizeof(float));
    cudaMalloc((void **)&d_C, n * k * sizeof(float));

    cudaMemcpy(d_A, h_A, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, m * k * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_A, d_B, d_C, n, m, k);

    cudaMemcpy(h_C, d_C, n * k * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("A = {\n");
    // print_mat((float *)h_A, n, m);
    // printf("}\n");
    // printf("B = {\n");
    // print_mat((float *)h_B, m, k);
    // printf("}\n");
    // printf("C = {\n");
    // print_mat((float *)h_C, n, k);
    // printf("}\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
