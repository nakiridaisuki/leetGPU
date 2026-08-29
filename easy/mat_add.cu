#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#define IDX_X (blockIdx.x * blockDim.x + threadIdx.x)
#define IDX_Y (blockIdx.y * blockDim.y + threadIdx.y)

__global__ void matrix_add(const float *A, const float *B, float *C, int N) {
    size_t idx = IDX_X;
    if (idx < N * N) {
        C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float *A, const float *B, float *C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
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
    const int n = 4096;
    size_t size = n * n * sizeof(float);
    srand(1000);

    float *A, *B, *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    for (int i = 0; i < n * n; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    solve(A, B, C, n);

    // printf("A = {\n");
    // print_mat((float *)A, n, n);
    // printf("}\n");
    // printf("B = {\n");
    // print_mat((float *)B, n, n);
    // printf("}\n");
    // printf("C = {\n");
    // print_mat((float *)C, n, n);
    // printf("}\n");

    for (int i = 0; i < n * n; i++) {
        if (C[i] != A[i] + B[i]) {
            printf("ERROR: %.2f != %.2f + %.2f\n", C[i], A[i], B[i]);
        }
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
