#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdio.h>

// Template parameters:
// R: Radix (2, 4, 8, etc.)
// N: Size of the FFT (must be a power of 2)

// Device function to compute the butterfly operation for radix-R FFT
template <int R>
__device__ void butterfly_fft(float2 v[R]) {
    for (int r = 0; r < R; ++r) {
        float2 sum = make_float2(0.0f, 0.0f);
        for (int m = 0; m < R; ++m) {
            float angle = -2.0f * CUDART_PI_F * r * m / R;
            float2 twiddle = make_float2(cosf(angle), sinf(angle));
            sum.x += v[m].x * twiddle.x - v[m].y * twiddle.y;
            sum.y += v[m].x * twiddle.y + v[m].y * twiddle.x;
        }
        v[r] = sum;
    }
}

// Device function to exchange data between threads using shared memory
template <int R, int N>
__device__ void exchange(float2 v[R], float2* sharedData, int tid, int s) {
    for (int r = 0; r < R; ++r) {
        int idx = (tid / (N / (R * s))) * (N / R) + (tid % (N / (R * s))) + r * (N / R);
        sharedData[idx] = v[r];
    }
    __syncthreads();

    for (int r = 0; r < R; ++r) {
        int idx = (tid / (N / (R * s))) * (N / R) + (tid % (N / (R * s))) + r * (N / R);
        v[r] = sharedData[idx];
    }
    __syncthreads();
}

// Device function to perform the FFT computation
template <int R, int N>
__device__ void fft(float2 v[R], float2* sharedData, int tid) {
    for (int s = 1; s < N; s *= R) {
        butterfly_fft<R>(v);  // Perform R-point FFT
        exchange<R, N>(v, sharedData, tid, s);  // Exchange data between threads
    }
}

// Main kernel to perform the shared memory FFT
template <int R, int N>
__global__ void fft_shared(float2* data) {
    extern __shared__ float2 sharedData[];  // Shared memory for data exchange
    int tid = threadIdx.x;  // Thread ID within the block
    int bid = blockIdx.x;   // Block ID

    // Local array for storing intermediate results
    float2 v[R];

    // Step 1: Load data from global memory to local array
    for (int r = 0; r < R; ++r) {
        int idx = bid * N + tid + r * (N / R);
        v[r] = data[idx];
    }

    // Step 2: Perform FFT computation
    fft<R, N>(v, sharedData, tid);

    // Step 3: Write data back to global memory
    for (int r = 0; r < R; ++r) {
        int idx = bid * N + tid + r * (N / R);
        data[idx] = v[r];
    }
}

// Host function to launch the kernel
template <int R, int N>
void launchSharedMemoryFFT(float2* d_data, int numFFTs) {
    int threadsPerBlock = N / R;  // One thread per R elements
    int blocksPerGrid = numFFTs;

    // Launch the kernel with shared memory size N * sizeof(float2)
    fft_shared<R, N><<<blocksPerGrid, threadsPerBlock, N * sizeof(float2)>>>(d_data);
}

int main() {
    const int R = 2;  // Radix (e.g., 2, 4, 8, etc.)
    const int N = 1024;  // FFT size (must be a power of 2)
    int numFFTs = 10;  // Number of FFTs to compute

    // Allocate memory on the device
    float2* d_data;
    cudaMalloc((void**)&d_data, N * numFFTs * sizeof(float2));

    // Launch the FFT kernel
    launchSharedMemoryFFT<R, N>(d_data, numFFTs);

    // Free memory
    cudaFree(d_data);

    return 0;
}