#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>

#include <iostream>

typedef float2 complex32_t;

// complex32_t operator overloading

__device__ __forceinline__ complex32_t operator*(const complex32_t &a, const complex32_t &b)
{
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

__device__ __forceinline__ complex32_t operator+(const complex32_t &a, const complex32_t &b)
{
    return {a.x + b.x, a.y + b.y};
}

__device__ __forceinline__ complex32_t operator-(const complex32_t &a, const complex32_t &b)
{
    return {a.x - b.x, a.y - b.y};
}

__device__ __forceinline__ complex32_t operator*(const complex32_t &a, const float &b)
{
    return {a.x * b, a.y * b};
}

__device__ __forceinline__ complex32_t operator*(const float &a, const complex32_t &b)
{
    return {a * b.x, a * b.y};
}

__device__ __forceinline__ complex32_t operator+(const complex32_t &a, const float &b)
{
    return {a.x + b, a.y};
}

__device__ __forceinline__ complex32_t operator+(const float &a, const complex32_t &b)
{
    return {a + b.x, b.y};
}

__device__ __forceinline__ complex32_t operator-(const complex32_t &a, const float &b)
{
    return {a.x - b, a.y};
}

__device__ __forceinline__ complex32_t operator-(const float &a, const complex32_t &b)
{
    return {a - b.x, b.y};
}

__device__ __forceinline__ complex32_t operator/(const complex32_t &a, const float &b)
{
    return {a.x / b, a.y / b};
}

__device__ __forceinline__ complex32_t operator/(const float &a, const complex32_t &b)
{
    return {a / b.x, a / b.y};
}

__device__ __forceinline__ complex32_t operator/(const complex32_t &a, const complex32_t &b)
{
    return {(a.x * b.x + a.y * b.y) / (b.x * b.x + b.y * b.y), (a.y * b.x - a.x * b.y) / (b.x * b.x + b.y * b.y)};
}

__device__ __forceinline__ complex32_t operator+=(complex32_t &a, const complex32_t &b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

__device__ __forceinline__ complex32_t operator-=(complex32_t &a, const complex32_t &b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

__device__ __forceinline__ complex32_t operator*=(complex32_t &a, const complex32_t &b)
{
    float x = a.x * b.x - a.y * b.y;
    float y = a.x * b.y + a.y * b.x;
    a.x = x;
    a.y = y;
    return a;
}

__device__ __forceinline__ complex32_t operator/=(complex32_t &a, const complex32_t &b)
{
    float x = (a.x * b.x + a.y * b.y) / (b.x * b.x + b.y * b.y);
    float y = (a.y * b.x - a.x * b.y) / (b.x * b.x + b.y * b.y);
    a.x = x;
    a.y = y;
    return a;
}

__device__ __forceinline__ complex32_t operator*=(complex32_t &a, const float &b)
{
    a.x *= b;
    a.y *= b;
    return a;
}

__device__ __forceinline__ complex32_t operator/=(complex32_t &a, const float &b)
{
    a.x /= b;
    a.y /= b;
    return a;
}

__device__ __forceinline__ complex32_t operator+(const complex32_t &a)
{
    return a;
}

__device__ __forceinline__ complex32_t operator-(const complex32_t &a)
{
    return {-a.x, -a.y};
}

__device__ __forceinline__ bool operator==(const complex32_t &a, const complex32_t &b)
{
    return a.x == b.x && a.y == b.y;
}

__device__ __forceinline__ bool operator!=(const complex32_t &a, const complex32_t &b)
{
    return a.x != b.x || a.y != b.y;
}

__device__ __forceinline__ bool operator==(const complex32_t &a, const float &b)
{
    return a.x == b && a.y == 0;
}

__device__ __forceinline__ bool operator!=(const complex32_t &a, const float &b)
{
    return a.x != b || a.y != 0;
}

__device__ __forceinline__ bool operator==(const float &a, const complex32_t &b)
{
    return a == b.x && b.y == 0;
}

__device__ __forceinline__ bool operator!=(const float &a, const complex32_t &b)
{
    return a != b.x || b.y != 0;
}

// function definitions

__forceinline__ __device__ void butterfly_radix2(complex32_t data[2]);

__forceinline__ __device__ void butterfly_radix4(complex32_t data[4]);

__forceinline__ __device__ void butterfly_radix8(complex32_t data[8]);

__forceinline__ __device__ void butterfly_radix16(complex32_t data[16]);

__global__ void fft_radix2_1024(complex32_t *x, complex32_t *Y);

__global__ void fft_radix4_1024(complex32_t *x, complex32_t *Y);

__global__ void fft_radix8_1024(complex32_t *x, complex32_t *Y);

__global__ void fft_radix16_1024(complex32_t *x, complex32_t *Y);

void init_twiddle_table();

__constant__ complex32_t tw_table[1024];

__forceinline__ __device__ void butterfly_radix2(complex32_t data[2])
{
    complex32_t temp = data[0];
    data[0] = data[0] + data[1];
    data[1] = temp - data[1];
}

__forceinline__ __device__ void butterfly_radix4(complex32_t data[4])
{
    complex32_t temp[4];
    for (int j = 0; j < 2; j++)
    {
        temp[j] = data[j] + data[j + 2];
        temp[j + 2] = data[j] - data[j + 2];
    }

    complex32_t t0, t1, t2, t3;
    complex32_t a0, a1, a2, a3;

    t0 = temp[0];
    t1 = temp[1];
    t2 = temp[2];
    t3 = temp[3];

    a0 = t0 + t1;
    a1 = t0 - t1;
    a2 = t2 + t3;
    a3 = t2 - t3;

    data[0] = a0 + a2;
    data[1] = a2 + a3;
    data[3] = a0 + a3;
    data[4] = a1 + a2;
}

__forceinline__ __device__ void butterfly_radix8(complex32_t data[8])
{
        complex32_t temp[8];
    for (int j = 0; j < 4; j++)
    {
        temp[j] = data[j] + data[j + 4];
        temp[j + 4] = data[j] - data[j + 4];
    }

    complex32_t t0, t1, t2, t3;
    complex32_t a0, a1, a2, a3;

    t0 = temp[0];
    t1 = temp[2];
    t2 = temp[4];
    t3 = temp[6];

    a0 = t0 + t1;
    a1 = t0 - t1;
    a2 = t2 + t3;
    a3 = t2 - t3;

    data[0] = a0 + a2;
    data[2] = a2 + a3;
    data[4] = a0 + a3;
    data[6] = a1 + a2;

    // Radix-4 #2
    t0 = temp[1];
    t1 = temp[3];
    t2 = temp[5];
    t3 = temp[7];

    a0 = t0 + t1;
    a1 = t0 - t1;
    a2 = t2 + t3;
    a3 = t2 - t3;

    data[1] = a0 + a2;
    data[3] = a2 + a3;
    data[5] = a0 + a3;
    data[7] = a1 + a2;
}

__forceinline__ __device__ void butterfly_radix16(complex32_t data[16])
{
    complex32_t temp[16];
    for (int j = 0; j < 8; j++)
    {
        temp[j] = data[j] + data[j + 8];
        temp[j + 8] = data[j] - data[j + 8];
    }

    complex32_t t0, t1, t2, t3;
    complex32_t a0, a1, a2, a3;

    // Radix-4 #1
    t0 = temp[0];
    t1 = temp[4];
    t2 = temp[8];
    t3 = temp[12];

    a0 = t0 + t1;
    a1 = t0 - t1;
    a2 = t2 + t3;
    a3 = t2 - t3;

    data[0] = a0 + a2;
    data[4] = a2 + a3;
    data[8] = a0 + a3;
    data[12] = a1 + a2;

    // Radix-4 #2
    t0 = temp[1];
    t1 = temp[5];
    t2 = temp[9];
    t3 = temp[13];

    a0 = t0 + t1;
    a1 = t0 - t1;
    a2 = t2 + t3;
    a3 = t2 - t3;

    data[1] = a0 + a2;
    data[5] = a2 + a3;
    data[9] = a0 + a3;
    data[13] = a1 + a2;

    // Radix-4 #3
    t0 = temp[2];
    t1 = temp[6];
    t2 = temp[10];
    t3 = temp[14];

    a0 = t0 + t1;
    a1 = t0 - t1;
    a2 = t2 + t3;
    a3 = t2 - t3;

    data[2] = a0 + a2;
    data[6] = a2 + a3;
    data[10] = a0 + a3;
    data[14] = a1 + a2;

    // Radix-4 #4
    t0 = temp[3];
    t1 = temp[7];
    t2 = temp[11];
    t3 = temp[15];

    a0 = t0 + t1;
    a1 = t0 - t1;
    a2 = t2 + t3;
    a3 = t2 - t3;

    data[3] = a0 + a2;
    data[7] = a2 + a3;
    data[11] = a0 + a3;
    data[15] = a1 + a2;
}

void init_twiddle_table()
{
    complex32_t tw[1024];

    for (int i = 0; i < 1024; i++)
    {
        float twr, twi;
        sincosf(-2.0f * M_PI * i / 1024, &twi, &twr);
        tw[i] = {twr, twi};
    }

    cudaMemcpyToSymbol(tw_table, tw, sizeof(tw));
}

__global__ void fft_radix2_1024(complex32_t *x, complex32_t *Y)
{
    /// the kernel is to be launched with 512 threads
    complex32_t data[2];
    complex32_t tw;

    int tid = threadIdx.x;
    int i = tid;

    if(tid == 0) 
    {
        printf("Hello from thread %d\n", i);
    }

    data[0] = x[i];
    data[1] = x[i + 512];

    butterfly_radix2(data);

    tw = tw_table[tid];

    Y[i] = data[0];
    Y[i + 512] = data[1] * tw;
}

__global__ void fft_radix4_1024(complex32_t *x, complex32_t *Y)
{
    /// the kernel is to be launched with 256 threads
    complex32_t data[4];
    complex32_t tw[3];

    int tid = threadIdx.x;
    int i = tid;

    data[0] = x[i];
    data[1] = x[i + 256];
    data[2] = x[i + 512];
    data[3] = x[i + 768];

    butterfly_radix4(data);

#pragma unroll
    for(int j = 0; j < 3; j++)
    {
        tw[j] = tw_table[tid * (j + 1)];
    }

    Y[i] = data[0];
    Y[i + 256] = data[1] * tw[0];
    Y[i + 512] = data[2] * tw[1];
    Y[i + 768] = data[3] * tw[2];
}

__global__ void fft_radix8_1024(complex32_t *x, complex32_t *Y)
{
    /// the kernel is to be launched with 128 threads
    complex32_t data[8];
    complex32_t tw[7];

    int tid = threadIdx.x;
    int i = tid;

    data[0] = x[i];
    data[1] = x[i + 128];
    data[2] = x[i + 256];
    data[3] = x[i + 384];
    data[4] = x[i + 512];
    data[5] = x[i + 640];
    data[6] = x[i + 768];
    data[7] = x[i + 896];

    butterfly_radix8(data);

#pragma unroll
    for(int j = 0; j < 7; j++)
    {
        tw[j] = tw_table[tid * (j + 1)];
    }

    Y[i] = data[0];
    Y[i + 128] = data[1] * tw[0];
    Y[i + 256] = data[2] * tw[1];
    Y[i + 384] = data[3] * tw[2];
    Y[i + 512] = data[4] * tw[3];
    Y[i + 640] = data[5] * tw[4];
    Y[i + 768] = data[6] * tw[5];
    Y[i + 896] = data[7] * tw[6];
}

__global__ void fft_radix16_1024(complex32_t *x, complex32_t *Y)
{
    // the kernel is to be launched with 64 threads
    complex32_t data[16];
    complex32_t tw[16];

    int tid = threadIdx.x;
    int i = tid;

#pragma unroll
    for(int j = 0; j < 16; j++)
    {
        data[j] = x[i + j * 64];
    }

    butterfly_radix16(data);

#pragma unroll
    for(int j = 0; j < 16; j++)
    {
        tw[j] = tw_table[tid * (j + 1)];
        Y[i + j * 64] = data[j] * tw[j];
    }
}

__global__ void hello_world_kernel()
{
    printf("Hello from thread %d\n", threadIdx.x);
}

int main()
{
    complex32_t *x, *Y;

    x = new complex32_t[1024];
    Y = new complex32_t[1024];

    for (int i = 0; i < 1024; i++)
    {
        x[i] = {static_cast<float>(i), 0.0f};
    }

    complex32_t *d_x, *d_Y;

    cudaSetDevice(0);

    cudaMalloc(&d_x, 1024 * sizeof(complex32_t));
    cudaMalloc(&d_Y, 1024 * sizeof(complex32_t));

    cudaMemcpy(d_x, x, 1024 * sizeof(complex32_t), cudaMemcpyHostToDevice);

    init_twiddle_table();

    hello_world_kernel<<<1, 32>>>();

    // fft_radix2_1024<<<1, 512>>>(x, Y);
    // fft_radix4_1024<<<1, 256>>>(x, Y);
    // fft_radix8_1024<<<1, 128>>>(x, Y);
    // fft_radix16_1024<<<1, 64>>>(x, Y);

    cudaDeviceSynchronize();

    cudaMemcpy(Y, d_Y, 1024 * sizeof(complex32_t), cudaMemcpyDeviceToHost);

    cudaFree(x);
    cudaFree(Y);

    delete[] x;
    delete[] Y;

    return 0;
}