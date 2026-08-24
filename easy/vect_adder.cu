#include <cuda_runtime.h>
#include <iostream>

__global__ void vector_add(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float *A, const float *B, float *C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

int main() {
    int n       = 5;
    size_t size = n * sizeof(float);

    float h_A[5] = {1, 2, 3, 4, 5};
    float h_B[5] = {6, 7, 8, 9, 10};
    float h_C[5];
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    solve(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result of C = { ");
    for (size_t i = 0; i < n; i++) {
        printf("%.2f, ", h_C[i]);
    }
    printf("}\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
