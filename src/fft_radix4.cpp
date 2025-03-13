#include "fft_radix4.h"
#include "bit_reverse.h"

#include <iostream>
#include <cmath>

void fft_radix4(complex32_t *x, complex32_t *Y, int n)
{
    if ((n & (n - 1)) != 0 || (static_cast<int>(std::log2(n)) % 2 != 0))
    {
        std::cerr << "Error: n must be a power of 4." << std::endl;
        return; // Or throw an exception
    }

    int log4N = std::log2(n) / 2;

    for (int i = 0; i < n; i++)
    {
        Y[bit_reverse(i, log4N)] = x[i];
    }

    for (int s = 2; s <= log4N; s++)
    {
        int m = 1 << s;
        int mh = m >> 2;

        for (int i = 0; i < n / 4; i++)
        {
            int k = (i / mh) * m;
            int j = i % mh;
            int kj = k + j;

            float twr, twi;
            __sincosf(-M_PI * j / mh, &twi, &twr);

            complex32_t tw = complex32_t(twr, twi);

            complex32_t u = Y[kj];
            complex32_t v = Y[kj + mh] * tw;
            complex32_t w = Y[kj + 2 * mh] * tw * tw;
            complex32_t t = Y[kj + 3 * mh] * tw * tw * tw;

            Y[kj] = u + v + w + t;
            Y[kj + mh] = u - v + complex32_t(0.0f, 1.0f) * w - complex32_t(0.0f, 1.0f) * t;
            Y[kj + 2 * mh] = u + v - w - t;
            Y[kj + 3 * mh] = u - v - complex32_t(0.0f, 1.0f) * w + complex32_t(0.0f, 1.0f) * t;
        }
    }
}

void fft_radix4_1024_v1(complex32_t *x, complex32_t *Y)
{
    constexpr int n = 1024;
    constexpr int n4 = n / 4;
    constexpr int n8 = n / 8;
    constexpr int log4n = 5;

    for (int i = 0; i < n4; i += 4)
    {
        Y[bit_reverse(i, log4n)] = x[i];
        Y[bit_reverse(i + 1, log4n)] = x[i + 1];
        Y[bit_reverse(i + 2, log4n)] = x[i + 2];
        Y[bit_reverse(i + 3, log4n)] = x[i + 3];
    }

    for (int s = 2; s <= log4n; s++)
    {
        int m = 1 << s;
        int mh = m >> 2;

        for (int i = 0; i < n8; i += 2)
        {
            int k = (i / mh) * m;
            int j = i % mh;
            int kj = k + j;

            float twr, twi;
            __sincosf(-M_PI * j / mh, &twi, &twr);

            complex32_t tw = complex32_t(twr, twi);
            complex32_t tw2 = tw * tw;
            complex32_t tw3 = tw2 * tw;

            complex32_t u = Y[kj];
            complex32_t v = Y[kj + mh] * tw;
            complex32_t w = Y[kj + 2 * mh] * tw2;
            complex32_t t = Y[kj + 3 * mh] * tw3;

            complex32_t v0, v1, v3, v4;
            complex32_t t0, t1, t3, t4;

            v0 = v + t;
            v1 = u - w;
            v3 = u + w;
            v4 = v - t;

            t0 = v0 + v3;
            t1 = v1 + v4;
            t3 = v1 - v4;
            t4 = v0 - v3;

            Y[kj] = t0;
            Y[kj + mh] = t1;
            Y[kj + 2 * mh] = t3;
            Y[kj + 3 * mh] = t4;

            // unravel the loop
            k = (i + 1) / mh * m;
            j = (i + 1) % mh;

            kj = k + j;

            __sincosf(-M_PI * j / mh, &twi, &twr);

            tw = complex32_t(twr, twi);
            tw2 = tw * tw;
            tw3 = tw2 * tw;

            u = Y[kj];
            v = Y[kj + mh] * tw;
            w = Y[kj + 2 * mh] * tw2;
            t = Y[kj + 3 * mh] * tw3;

            v0 = v + t;
            v1 = u - w;
            v3 = u + w;
            v4 = v - t;

            t0 = v0 + v3;
            t1 = v1 + v4;
            t3 = v1 - v4;
            t4 = v0 - v3;

            Y[kj] = t0;
            Y[kj + mh] = t1;
            Y[kj + 2 * mh] = t3;
            Y[kj + 3 * mh] = t4;
        }
    }
}

void fft_radix4_no_bit_reverse(complex32_t *x, complex32_t *Y, int n)
{
    if ((n & (n - 1)) != 0 || (static_cast<int>(std::log2(n)) % 2 != 0))
    {
        std::cerr << "Error: n must be a power of 4." << std::endl;
        return;
    }

    // Copy input to output for the first stage.
    for (int i = 0; i < n; i++)
    {
        Y[i] = x[i];
    }

    int log4N = std::log2(n) / 2;

    for (int s = 1; s <= log4N; s++)
    {
        int m = 1 << (2 * s); // m = 4^s
        int mh = m >> 2;      // mh = m/4

        for (int k = 0; k < n; k += m)
        { 
            // Iterate over groups of size m
            for (int j = 0; j < mh; j++)
            { 
                // Iterate within each group's sub-butterflies
                float twr, twi;
                __sincosf(-2.0f * M_PI * j / m, &twi, &twr); // Correct twiddle factor

                complex32_t tw = complex32_t(twr, twi);

                complex32_t data[4];
                data[0] = Y[k + j];
                data[1] = Y[k + j + mh] * tw;
                data[2] = Y[k + j + 2 * mh] * tw * tw;
                data[3] = Y[k + j + 3 * mh] * tw * tw * tw;

                Y[k + j] = data[0] + data[1] + data[2] + data[3];
                Y[k + j + mh] = data[0] - data[1] + complex32_t(0.0f, 1.0f) * data[2] - complex32_t(0.0f, 1.0f) * data[3];
                Y[k + j + 2 * mh] = data[0] + data[1] - data[2] - data[3];
                Y[k + j + 3 * mh] = data[0] - data[1] - complex32_t(0.0f, 1.0f) * data[2] + complex32_t(0.0f, 1.0f) * data[3];
            }
        }
    }
}