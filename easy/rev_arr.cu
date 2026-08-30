#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#define IDX_X (blockIdx.x * blockDim.x + threadIdx.x)

__global__ void reverse_array(float *input, int N) {
    int idx = IDX_X;
    if (idx < N / 2) {
        float tmp          = input[idx];
        input[idx]         = input[N - 1 - idx];
        input[N - 1 - idx] = tmp;
    }
}

__global__ void reverse_array_f4(float4 *input, int N) {
    int idx = IDX_X;
    if (idx < N / 2) {
        float4 left  = input[idx];
        float4 right = input[N - 1 - idx];

        float4 new_left  = make_float4(left.w, left.z, left.y, left.x);
        float4 new_right = make_float4(right.w, right.z, right.y, right.x);

        input[idx]         = new_right;
        input[N - 1 - idx] = new_left;
    }
}
// input is device pointer
void solve(float *input, int N) {
    int threadsPerBlock  = 256;
    int blocksPerGrid    = (N / 2 + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid_f4 = (N / 8 + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < 5; i++)
        reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    for (int i = 0; i < 5; i++)
        reverse_array_f4<<<blocksPerGrid_f4, threadsPerBlock>>>(
            (float4 *)input, N / 4
        );
    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
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
    const int n = 25000000;
    size_t size = n * sizeof(float);
    float *input, *output;
    input  = (float *)malloc(size);
    output = (float *)malloc(size);

    float *d_input;
    cudaMalloc((void **)&d_input, size);

    srand(1000);
    for (int i = 0; i < n; i++) {
        input[i] = rand() % 10;
    }

    cudaMemcpy(d_input, input, size, cudaMemcpyDefault);
    solve(d_input, n);
    cudaMemcpy(output, d_input, size, cudaMemcpyDefault);

    // printf("input = {\n");
    // print_mat((float *)input, 1, n);
    // printf("}\n");
    // printf("output = {\n");
    // print_mat((float *)output, 1, n);
    // printf("}\n");

    for (int i = 0; i < n; i++) {
        if (input[n - i - 1] != output[i]) {
            printf(
                "ERROR at idx %d, %.2f != %.2f\n",
                i,
                input[n - i - 1],
                output[i]
            );
        }
    }

    cudaFree(input);
    cudaFree(output);
}
