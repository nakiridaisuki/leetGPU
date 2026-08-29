#include <cuda_runtime.h>
#include <iostream>

#define IDX_X (blockIdx.x * blockDim.x + threadIdx.x)
const int BLOCK_DIM = 256;

__global__ void invert_kernel(unsigned char *image, int width, int height) {
    int idx   = IDX_X * 4;
    int total = width * height * 4;
    if (idx < total) {
        image[idx]     = 255 - image[idx];
        image[idx + 1] = 255 - image[idx + 1];
        image[idx + 2] = 255 - image[idx + 2];
    }
}

__global__ void
invert_kernel_optim(unsigned char *image, int width, int height) {
    int idx   = IDX_X;
    int total = width * height;
    if (idx < total) {
        uchar4 *image_ptr = (uchar4 *)image;

        uchar4 pixel = image_ptr[idx];

        pixel.x = 255 - pixel.x;
        pixel.y = 255 - pixel.y;
        pixel.z = 255 - pixel.z;

        image_ptr[idx] = pixel;
    }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the
// GPU)
void solve(unsigned char *image, int width, int height) {
    int threadsPerBlock = BLOCK_DIM;
    int blocksPerGrid =
        (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
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
    int width = 2, height = 3;
    size_t size = width * height * 4 * sizeof(char);
    srand(1000);

    unsigned char image[size];
    unsigned char result[size];
    for (int i = 0; i < size; i++) {
        image[i] = rand() % 256;
    }
    printf("Inout  = { ");
    for (int i = 0; i < size; i++)
        printf("%3d ", image[i]);
    printf("}\n");

    unsigned char *d_i;
    cudaMalloc((void **)&d_i, size);

    cudaMemcpy(d_i, image, size, cudaMemcpyHostToDevice);

    solve(d_i, height, width);

    cudaMemcpy(result, d_i, size, cudaMemcpyDeviceToHost);
    printf("Output = { ");
    for (int i = 0; i < size; i++)
        printf("%3d ", result[i]);
    printf("}\n");
    for (int i = 0; i < size; i++) {
        if (abs(image[i] - result[i]) > 255) {
            printf("ERROR\n");
        }
    }

    cudaFree(d_i);
}

void profile() {
    int width = 5120, height = 4096;
    size_t size = width * height * 4 * sizeof(char);
    srand(1000);

    unsigned char *image;
    cudaMalloc((void **)&image, size);
    cudaMemset(image, 1, size); // 初始化為1

    int threadsPerBlock = BLOCK_DIM;
    int blocksPerGrid =
        (width * height + threadsPerBlock - 1) / threadsPerBlock;

    // 準備計時器
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_naive, ms_opt;

    // --- 暖身 (Warmup) ---
    // 第一次執行通常會包含延遲，所以先各跑一次不計時
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    invert_kernel_optim<<<blocksPerGrid, threadsPerBlock>>>(
        image, width, height
    );
    cudaDeviceSynchronize();

    // --- 測試 Naive 版本 ---
    cudaEventRecord(start);
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_naive, start, stop);
    printf("Naive Kernel Time:     %.3f ms\n", ms_naive);

    // --- 測試 Optimized 版本 ---
    cudaEventRecord(start);
    invert_kernel_optim<<<blocksPerGrid, threadsPerBlock>>>(
        image, width, height
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_opt, start, stop);
    printf("Optimized Kernel Time: %.3f ms\n", ms_opt);
    printf("Speedup: %.2f x\n", ms_naive / ms_opt);

    // 清理資源
    cudaFree(image);
}

int main() {
    // test();
    profile();
}
