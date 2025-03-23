__global__ void fft_radix2_1024(float2* data) {
    extern __shared__ float2 sharedData[];  // Shared memory for data exchange
    int tid = threadIdx.x;  // Thread ID within the block
    int bid = blockIdx.x;   // Block ID

    // Local array for storing intermediate results
    float2 v[2];

    // Step 1: Load data from global memory to local array
    for (int r = 0; r < 2; ++r) {
        int idx = bid * 1024 + tid + r * 512;
        v[r] = data[idx];
    }

    // Step 2: Perform FFT in local array and exchange data via shared memory
    for (int s = 1; s < 1024; s *= 2) {
        // Perform radix-2 butterfly
        butterfly_radix2(v[0], v[1]);

        // Exchange data between threads using shared memory
        int idx1 = (tid / s) * (2 * s) + (tid % s);
        int idx2 = idx1 + s;
        sharedData[idx1] = v[0];
        sharedData[idx2] = v[1];
        __syncthreads();

        // Load exchanged data back into local array
        v[0] = sharedData[idx1];
        v[1] = sharedData[idx2];
        __syncthreads();
    }

    // Step 3: Write data back to global memory
    for (int r = 0; r < 2; ++r) {
        int idx = bid * 1024 + tid + r * 512;
        data[idx] = v[r];
    }
}

template <>
__global__ void fft_radix4_1024(float2* data) {
    extern __shared__ float2 sharedData[];  // Shared memory for data exchange
    int tid = threadIdx.x;  // Thread ID within the block
    int bid = blockIdx.x;   // Block ID

    // Local array for storing intermediate results
    float2 v[4];

    // Step 1: Load data from global memory to local array
    for (int r = 0; r < 4; ++r) {
        int idx = bid * 1024 + tid + r * 256;
        v[r] = data[idx];
    }

    // Step 2: Perform FFT in local array and exchange data via shared memory
    for (int s = 1; s < 1024; s *= 4) {
        // Perform radix-4 butterfly
        butterfly_radix4(v);

        // Exchange data between threads using shared memory
        int idx = (tid / s) * (4 * s) + (tid % s);
        for (int r = 0; r < 4; ++r) {
            sharedData[idx + r * s] = v[r];
        }
        __syncthreads();

        // Load exchanged data back into local array
        for (int r = 0; r < 4; ++r) {
            v[r] = sharedData[idx + r * s];
        }
        __syncthreads();
    }

    // Step 3: Write data back to global memory
    for (int r = 0; r < 4; ++r) {
        int idx = bid * 1024 + tid + r * 256;
        data[idx] = v[r];
    }
}

__global__ void fft_radix8_1024(float2* data) {
    extern __shared__ float2 sharedData[];  // Shared memory for data exchange
    int tid = threadIdx.x;  // Thread ID within the block
    int bid = blockIdx.x;   // Block ID

    // Local array for storing intermediate results (8 elements per thread)
    float2 v[8];

    // Step 1: Load data from global memory to local array
    for (int r = 0; r < 8; ++r) {
        int idx = bid * 1024 + tid + r * 128;
        v[r] = data[idx];
    }

    // Step 2: Perform 3 stages of radix-8 FFT
    for (int s = 1; s < 512; s *= 8) {
        // Perform radix-8 butterfly
        butterfly_radix8(v);

        // Exchange data between threads using shared memory
        int idx = (tid / s) * (8 * s) + (tid % s);
        for (int r = 0; r < 8; ++r) {
            sharedData[idx + r * s] = v[r];
        }
        __syncthreads();

        // Load exchanged data back into local array
        for (int r = 0; r < 8; ++r) {
            v[r] = sharedData[idx + r * s];
        }
        __syncthreads();
    }

    // Step 3: Perform 1 stage of radix-2 FFT
    // Each thread computes 4 radix-2 butterflies
    for (int i = 0; i < 4; ++i) {
        float2 v2[2];
        for (int r = 0; r < 2; ++r) {
            int idx = i * 2 + r;
            v2[r] = v[idx];
        }
        butterfly_radix2(v2[0], v2[1]);  // Perform radix-2 butterfly
        for (int r = 0; r < 2; ++r) {
            int idx = i * 2 + r;
            v[idx] = v2[r];
        }
    }

    // Step 4: Write data back to global memory
    for (int r = 0; r < 8; ++r) {
        int idx = bid * 1024 + tid + r * 128;
        data[idx] = v[r];
    }
}

__global__ void fft_radix16_1024(float2* data) {
    extern __shared__ float2 sharedData[];  // Shared memory for data exchange
    int tid = threadIdx.x;  // Thread ID within the block
    int bid = blockIdx.x;   // Block ID

    // Local array for storing intermediate results (16 elements per thread)
    float2 v[16];

    // Step 1: Load data from global memory to local array
    for (int r = 0; r < 16; ++r) {
        int idx = bid * 1024 + tid + r * 64;
        v[r] = data[idx];
    }

    // Step 2: Perform 2 stages of radix-16 FFT
    for (int s = 1; s < 256; s *= 16) {
        // Perform radix-16 butterfly
        butterfly_radix16(v);

        // Exchange data between threads using shared memory
        int idx = (tid / s) * (16 * s) + (tid % s);
        for (int r = 0; r < 16; ++r) {
            sharedData[idx + r * s] = v[r];
        }
        __syncthreads();

        // Load exchanged data back into local array
        for (int r = 0; r < 16; ++r) {
            v[r] = sharedData[idx + r * s];
        }
        __syncthreads();
    }

    // Step 3: Perform 1 stage of radix-4 FFT
    // Each thread computes 4 radix-4 butterflies
    for (int i = 0; i < 4; ++i) {
        float2 v4[4];
        for (int r = 0; r < 4; ++r) {
            int idx = i * 4 + r;
            v4[r] = v[idx];
        }
        butterfly_radix4(v4);  // Perform radix-4 butterfly
        for (int r = 0; r < 4; ++r) {
            int idx = i * 4 + r;
            v[idx] = v4[r];
        }
    }

    // Step 4: Write data back to global memory
    for (int r = 0; r < 16; ++r) {
        int idx = bid * 1024 + tid + r * 64;
        data[idx] = v[r];
    }
}

__device__ void butterfly_radix2(float2& a, float2& b, const float2& twiddle) {
    float2 t = a;
    a = t + b * twiddle;  // Apply twiddle factor
    b = t - b * twiddle;  // Apply twiddle factor
}

__device__ void butterfly_radix4(float2 v[4], const float2 twiddles[4]) {
    float2 t0 = v[0];
    float2 t1 = v[1] * twiddles[1];  // Apply twiddle factor
    float2 t2 = v[2] * twiddles[2];  // Apply twiddle factor
    float2 t3 = v[3] * twiddles[3];  // Apply twiddle factor

    // Perform radix-4 butterfly
    v[0] = t0 + t1 + t2 + t3;
    v[1] = t0 - t1 + t2 - t3;
    v[2] = t0 + t1 - t2 - t3;
    v[3] = t0 - t1 - t2 + t3;
}

__device__ void butterfly_radix8(float2 v[8], const float2 twiddles[8]) {
    float2 t[8];
    for (int r = 0; r < 8; ++r) {
        t[r] = v[r] * twiddles[r];  // Apply twiddle factors
    }

    // Perform radix-8 butterfly
    v[0] = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
    v[1] = t[0] - t[1] + t[2] - t[3] + t[4] - t[5] + t[6] - t[7];
    v[2] = t[0] + t[1] - t[2] - t[3] + t[4] + t[5] - t[6] - t[7];
    v[3] = t[0] - t[1] - t[2] + t[3] + t[4] - t[5] - t[6] + t[7];
    v[4] = t[0] + t[1] + t[2] + t[3] - t[4] - t[5] - t[6] - t[7];
    v[5] = t[0] - t[1] + t[2] - t[3] - t[4] + t[5] - t[6] + t[7];
    v[6] = t[0] + t[1] - t[2] - t[3] - t[4] - t[5] + t[6] + t[7];
    v[7] = t[0] - t[1] - t[2] + t[3] - t[4] + t[5] + t[6] - t[7];
}

__device__ void butterfly_radix16(float2 v[16], const float2 twiddles[16]) {
    float2 t[16];
    for (int r = 0; r < 16; ++r) {
        t[r] = v[r] * twiddles[r];  // Apply twiddle factors
    }

    // Perform radix-16 butterfly
    v[0] = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7] + t[8] + t[9] + t[10] + t[11] + t[12] + t[13] + t[14] + t[15];
    v[1] = t[0] - t[1] + t[2] - t[3] + t[4] - t[5] + t[6] - t[7] + t[8] - t[9] + t[10] - t[11] + t[12] - t[13] + t[14] - t[15];
    v[2] = t[0] + t[1] - t[2] - t[3] + t[4] + t[5] - t[6] - t[7] + t[8] + t[9] - t[10] - t[11] + t[12] + t[13] - t[14] - t[15];
    v[3] = t[0] - t[1] - t[2] + t[3] + t[4] - t[5] - t[6] + t[7] + t[8] - t[9] - t[10] + t[11] + t[12] - t[13] - t[14] + t[15];
    v[4] = t[0] + t[1] + t[2] + t[3] - t[4] - t[5] - t[6] - t[7] + t[8] + t[9] + t[10] + t[11] - t[12] - t[13] - t[14] - t[15];
    v[5] = t[0] - t[1] + t[2] - t[3] - t[4] + t[5] - t[6] + t[7] + t[8] - t[9] + t[10] - t[11] - t[12] + t[13] - t[14] + t[15];
    v[6] = t[0] + t[1] - t[2] - t[3] - t[4] - t[5] + t[6] + t[7] + t[8] + t[9] - t[10] - t[11] - t[12] - t[13] + t[14] + t[15];
    v[7] = t[0] - t[1] - t[2] + t[3] - t[4] + t[5] + t[6] - t[7] + t[8] - t[9] - t[10] + t[11] - t[12] + t[13] + t[14] - t[15];
    v[8] = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7] - t[8] - t[9] - t[10] - t[11] - t[12] - t[13] - t[14] - t[15];
    v[9] = t[0] - t[1] + t[2] - t[3] + t[4] - t[5] + t[6] - t[7] - t[8] + t[9] - t[10] + t[11] - t[12] + t[13] - t[14] + t[15];
    v[10] = t[0] + t[1] - t[2] - t[3] + t[4] + t[5] - t[6] - t[7] - t[8] - t[9] + t[10] + t[11] - t[12] - t[13] + t[14] + t[15];
    v[11] = t[0] - t[1] - t[2] + t[3] + t[4] - t[5] - t[6] + t[7] - t[8] + t[9] + t[10] - t[11] - t[12] + t[13] + t[14] - t[15];
    v[12] = t[0] + t[1] + t[2] + t[3] - t[4] - t[5] - t[6] - t[7] - t[8] - t[9] - t[10] - t[11] + t[12] + t[13] + t[14] + t[15];
    v[13] = t[0] - t[1] + t[2] - t[3] - t[4] + t[5] - t[6] + t[7] - t[8] + t[9] - t[10] + t[11] + t[12] - t[13] + t[14] - t[15];
    v[14] = t[0] + t[1] - t[2] - t[3] - t[4] - t[5] + t[6] + t[7] - t[8] - t[9] + t[10] + t[11] + t[12] + t[13] - t[14] - t[15];
    v[15] = t[0] - t[1] - t[2] + t[3] - t[4] + t[5] + t[6] - t[7] - t[8] + t[9] + t[10] - t[11] + t[12] - t[13] - t[14] + t[15];
}

constexpr float2 twiddles_radix2[2] = {
    {1.0f, 0.0f},  // e^(0)
    {-1.0f, 0.0f}  // e^(-i*pi)
};

constexpr float2 twiddles_radix4[4] = {
    {1.0f, 0.0f},          // e^(0)
    {0.0f, -1.0f},         // e^(-i*pi/2)
    {-1.0f, 0.0f},         // e^(-i*pi)
    {0.0f, 1.0f}           // e^(-i*3pi/2)
};

constexpr float2 twiddles_radix8[8] = {
    {1.0f, 0.0f},                          // e^(0)
    {0.70710678118f, -0.70710678118f},     // e^(-i*pi/4)
    {0.0f, -1.0f},                         // e^(-i*pi/2)
    {-0.70710678118f, -0.70710678118f},    // e^(-i*3pi/4)
    {-1.0f, 0.0f},                         // e^(-i*pi)
    {-0.70710678118f, 0.70710678118f},     // e^(-i*5pi/4)
    {0.0f, 1.0f},                          // e^(-i*3pi/2)
    {0.70710678118f, 0.70710678118f}       // e^(-i*7pi/4)
};

constexpr float2 twiddles_radix16[16] = {
    {1.0f, 0.0f},                          // e^(0)
    {0.92387953251f, -0.38268343236f},     // e^(-i*pi/8)
    {0.70710678118f, -0.70710678118f},     // e^(-i*pi/4)
    {0.38268343236f, -0.92387953251f},     // e^(-i*3pi/8)
    {0.0f, -1.0f},                         // e^(-i*pi/2)
    {-0.38268343236f, -0.92387953251f},    // e^(-i*5pi/8)
    {-0.70710678118f, -0.70710678118f},    // e^(-i*3pi/4)
    {-0.92387953251f, -0.38268343236f},    // e^(-i*7pi/8)
    {-1.0f, 0.0f},                         // e^(-i*pi)
    {-0.92387953251f, 0.38268343236f},     // e^(-i*9pi/8)
    {-0.70710678118f, 0.70710678118f},     // e^(-i*5pi/4)
    {-0.38268343236f, 0.92387953251f},     // e^(-i*11pi/8)
    {0.0f, 1.0f},                          // e^(-i*3pi/2)
    {0.38268343236f, 0.92387953251f},      // e^(-i*13pi/8)
    {0.70710678118f, 0.70710678118f},      // e^(-i*7pi/4)
    {0.92387953251f, 0.38268343236f}       // e^(-i*15pi/8)
};