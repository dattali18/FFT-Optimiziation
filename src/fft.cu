#include <cuda_runtime.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "complex.h"


// Precomputed twiddle factors for device
__constant__ complex32_t d_W1024[1024];

// Initialize twiddle factors on host and copy to device
void init_twiddles() {
    complex32_t h_W1024[1024];
    
    for (int k = 0; k < 1024; k++) {
        float angle = -2.0f * M_PI * k / 1024.0f;
        h_W1024[k].x = cosf(angle);
        h_W1024[k].y = sinf(angle);
    }
    
    cudaMemcpyToSymbol(d_W1024, h_W1024, sizeof(complex32_t) * 1024);
}

/**
 * Improved radix-16 butterfly operation for a single butterfly
 * This function performs a complete radix-16 butterfly operation on 16 elements
 */
__device__ void radix16_butterfly(complex32_t data[16]) {
    // First level: 8 radix-2 butterflies
    complex32_t temp[16];
    for (int j = 0; j < 8; j++) {
        temp[j] = complex_add(data[j], data[j + 8]);
        temp[j + 8] = complex_sub(data[j], data[j + 8]);
    }
    
    // Second level: 4 radix-4 butterflies
    complex32_t t0, t1, t2, t3;
    
    // Radix-4 #1
    t0 = temp[0];
    t1 = temp[4];
    t2 = temp[8];
    t3 = temp[12];
    
    data[0] = complex_add(complex_add(t0, t1), complex_add(t2, t3));
    data[4] = complex_add(complex_sub(t0, t1), complex_sub(t2, t3));
    data[8] = complex_add(complex_add(t0, t1), complex_sub(t2, t3));
    data[12] = complex_add(complex_sub(t0, t1), complex_add(t2, t3));
    
    // Radix-4 #2
    t0 = temp[1];
    t1 = temp[5];
    t2 = temp[9];
    t3 = temp[13];
    
    data[1] = complex_add(complex_add(t0, t1), complex_add(t2, t3));
    data[5] = complex_add(complex_sub(t0, t1), complex_sub(t2, t3));
    data[9] = complex_add(complex_add(t0, t1), complex_sub(t2, t3));
    data[13] = complex_add(complex_sub(t0, t1), complex_add(t2, t3));
    
    // Radix-4 #3
    t0 = temp[2];
    t1 = temp[6];
    t2 = temp[10];
    t3 = temp[14];
    
    data[2] = complex_add(complex_add(t0, t1), complex_add(t2, t3));
    data[6] = complex_add(complex_sub(t0, t1), complex_sub(t2, t3));
    data[10] = complex_add(complex_add(t0, t1), complex_sub(t2, t3));
    data[14] = complex_add(complex_sub(t0, t1), complex_add(t2, t3));
    
    // Radix-4 #4
    t0 = temp[3];
    t1 = temp[7];
    t2 = temp[11];
    t3 = temp[15];
    
    data[3] = complex_add(complex_add(t0, t1), complex_add(t2, t3));
    data[7] = complex_add(complex_sub(t0, t1), complex_sub(t2, t3));
    data[11] = complex_add(complex_add(t0, t1), complex_sub(t2, t3));
    data[15] = complex_add(complex_sub(t0, t1), complex_add(t2, t3));
}

// Radix-16 FFT kernel for 1024-point transform with better thread utilization
__global__ void fft_radix16_kernel(complex32_t* input, complex32_t* output, int batch_size) {
    // Thread and block indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int thread_count = blockDim.x;
    
    // Only process valid batches
    if (bid >= batch_size) return;
    
    // Shared memory for the FFT data
    __shared__ complex32_t shared_data[1024];
    
    // Load data from global memory - each thread loads multiple elements
    // This approach works well even with many threads
    for (int i = tid; i < 1024; i += thread_count) {
        shared_data[i] = input[bid * 1024 + i];
    }
    
    __syncthreads();
    
    // Stage 1: 64 radix-16 butterflies
    // Multiple threads process different butterflies in parallel
    for (int butterfly_idx = tid; butterfly_idx < 64; butterfly_idx += thread_count) {
        // Load 16 points for this butterfly
        complex32_t data[16];
        for (int j = 0; j < 16; j++) {
            data[j] = shared_data[butterfly_idx * 16 + j];
        }
        
        // Perform radix-16 butterfly
        radix16_butterfly(data);
        
        // Store results with stride for next stage
        for (int j = 0; j < 16; j++) {
            shared_data[j * 64 + butterfly_idx] = data[j];
        }
    }
    
    __syncthreads();
    
    // Stage 2: 16 radix-16 butterflies
    // Multiple threads process different butterflies in parallel
    for (int butterfly_idx = tid; butterfly_idx < 16; butterfly_idx += thread_count) {
        // Process one radix-16 butterfly
        complex32_t data[16];
        
        // Load 16 points with stride
        for (int j = 0; j < 16; j++) {
            data[j] = shared_data[butterfly_idx + j * 16];
        }
        
        // Apply twiddle factors for this stage
        for (int j = 1; j < 16; j++) {
            int twiddle_idx = (butterfly_idx * j * 64) % 1024;
            data[j] = complex_mul(data[j], d_W1024[twiddle_idx]);
        }
        
        // Perform butterfly operation
        radix16_butterfly(data);
        
        // Store results with natural ordering
        for (int j = 0; j < 16; j++) {
            int idx = butterfly_idx * 64 + j * 4;
            shared_data[idx] = data[j];
        }
    }
    
    __syncthreads();
    
    // Final stage: Radix-4 operations
    // Multiple threads process different radix-4 butterflies
    for (int r4_idx = tid; r4_idx < 256; r4_idx += thread_count) {
        // Load 4 points
        complex32_t a0 = shared_data[r4_idx];
        complex32_t a1 = shared_data[r4_idx + 256];
        complex32_t a2 = shared_data[r4_idx + 512];
        complex32_t a3 = shared_data[r4_idx + 768];
        
        // Apply twiddle factors
        int tw1 = (r4_idx * 1) % 1024;
        int tw2 = (r4_idx * 2) % 1024;
        int tw3 = (r4_idx * 3) % 1024;
        
        a1 = complex_mul(a1, d_W1024[tw1]);
        a2 = complex_mul(a2, d_W1024[tw2]);
        a3 = complex_mul(a3, d_W1024[tw3]);
        
        // Radix-4 butterfly
        complex32_t res0 = complex_add(complex_add(a0, a1), complex_add(a2, a3));
        complex32_t res1 = complex_add(complex_sub(a0, a1), complex_mul(complex_sub(a2, a3), d_W1024[256]));
        complex32_t res2 = complex_add(complex_sub(a0, a2), complex_mul(complex_sub(a1, a3), d_W1024[512]));
        complex32_t res3 = complex_add(complex_sub(a0, a3), complex_mul(complex_sub(a1, a2), d_W1024[768]));
        
        // Store results
        shared_data[r4_idx] = res0;
        shared_data[r4_idx + 256] = res1;
        shared_data[r4_idx + 512] = res2;
        shared_data[r4_idx + 768] = res3;
    }
    
    __syncthreads();
    
    // Copy results back to global memory
    for (int i = tid; i < 1024; i += thread_count) {
        output[bid * 1024 + i] = shared_data[i];
    }
}

// Host wrapper function
void fft_kernel(complex32_t* input, complex32_t* output, int batch_size) {
    static bool twiddles_initialized = false;
    
    // Initialize twiddle factors if needed
    if (!twiddles_initialized) {
        init_twiddles();
        twiddles_initialized = true;
    }
    
    // Launch kernel with appropriate grid and block dimensions
    // Using more threads per block for better occupancy
    int threads_per_block = 256; // Typical sweet spot for many GPUs
    int blocks = batch_size;
    
    fft_radix16_kernel<<<blocks, threads_per_block>>>(input, output, batch_size);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
    }
}

// Example usage
int main() {
    // Example with a batch of 10 FFTs
    const int batch_size = 10;
    const int fft_size = 1024;
    
    // Allocate host memory
    complex32_t* h_input = (complex32_t*)malloc(batch_size * fft_size * sizeof(complex32_t));
    complex32_t* h_output = (complex32_t*)malloc(batch_size * fft_size * sizeof(complex32_t));
    
    // Initialize input data
    for (int i = 0; i < batch_size * fft_size; i++) {
        h_input[i].x = (float)(i % 1024) / 1024.0f;
        h_input[i].y = 0.0f;
    }
    
    // Allocate device memory
    complex32_t* d_input;
    complex32_t* d_output;
    
    cudaMalloc(&d_input, batch_size * fft_size * sizeof(complex32_t));
    cudaMalloc(&d_output, batch_size * fft_size * sizeof(complex32_t));
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, batch_size * fft_size * sizeof(complex32_t), 
               cudaMemcpyHostToDevice);
    
    // Execute FFT
    fft_kernel(d_input, d_output, batch_size);
    
    // Copy results back to host
    cudaMemcpy(h_output, d_output, batch_size * fft_size * sizeof(complex32_t), 
               cudaMemcpyDeviceToHost);
    
    // Clean up
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}