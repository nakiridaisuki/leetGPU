#include <cuda_runtime.h>
#include <iostream>

#define IDX_X (blockIdx.x * blockDim.x + threadIdx.x)
#define IDX_Y (blockIdx.y * blockDim.y + threadIdx.y)
const int TILE_DIM = 16;

__global__ void
matrix_transpose_kernel(const float *input, float *output, int rows, int cols) {
    int i = IDX_X, j = IDX_Y;
    if (i < cols && j < rows) {
        output[i * rows + j] = input[j * cols + i];
    }
}

__global__ void matrix_transpose_optimized(
    const float *input, float *output, int rows, int cols
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];

    __syncthreads();

    int x_out = blockIdx.y * TILE_DIM + threadIdx.x;
    int y_out = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x_out < rows && y_out < cols)
        output[y_out * rows + x_out] = tile[threadIdx.x][threadIdx.y];
}

void solve(const float *input, float *output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    //     input, output, rows, cols
    // );
    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, rows, cols
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

void test() {
    int col = 2, row = 3;
    size_t size = col * row * sizeof(float);
    srand(1000);

    float input[row][col];
    float output[col][row];
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            input[i][j] = rand() % 10;
        }
    }
    float *d_i, *d_o;
    cudaMalloc((void **)&d_i, size);
    cudaMalloc((void **)&d_o, size);

    cudaMemcpy(d_i, input, size, cudaMemcpyHostToDevice);

    solve(d_i, d_o, row, col);

    cudaMemcpy(output, d_o, size, cudaMemcpyDeviceToHost);

    printf("Inout = {\n");
    print_mat((float *)input, row, col);
    printf("}\n");
    printf("Output = {\n");
    print_mat((float *)output, col, row);
    printf("}\n");

    cudaFree(d_i);
    cudaFree(d_o);
}

void profile() {
    int col = 8192, row = 8192;
    size_t size = col * row * sizeof(float);
    srand(1000);

    float *d_i, *d_o;
    cudaMalloc((void **)&d_i, size);
    cudaMalloc((void **)&d_o, size);
    cudaMemset(d_i, 1, size); // 初始化為1

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid(
        (col + TILE_DIM - 1) / TILE_DIM, (row + TILE_DIM - 1) / TILE_DIM
    );

    // 準備計時器
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_naive, ms_opt;

    // --- 暖身 (Warmup) ---
    // 第一次執行通常會包含延遲，所以先各跑一次不計時
    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_i, d_o, row, col
    );
    matrix_transpose_optimized<<<blocksPerGrid, threadsPerBlock>>>(
        d_i, d_o, row, col
    );
    cudaDeviceSynchronize();

    // --- 測試 Naive 版本 ---
    cudaEventRecord(start);
    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_i, d_o, row, col
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_naive, start, stop);
    printf("Naive Kernel Time:     %.3f ms\n", ms_naive);

    // --- 測試 Optimized 版本 ---
    cudaEventRecord(start);
    matrix_transpose_optimized<<<blocksPerGrid, threadsPerBlock>>>(
        d_i, d_o, row, col
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_opt, start, stop);
    printf("Optimized Kernel Time: %.3f ms\n", ms_opt);
    printf("Speedup: %.2f x\n", ms_naive / ms_opt);

    // 清理資源
    cudaFree(d_i);
    cudaFree(d_o);
}

int main() {
    // test();
    profile();
}
