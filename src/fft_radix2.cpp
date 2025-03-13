#include "fft_radix2.h"
#include "bit_reverse.h"

void fft_radix2(complex32_t *x, complex32_t *Y, int n)
{
    int logN = std::log2(n);

    // bit reversal permutation
    for (int i = 0; i < n; i++)
    {
        Y[bit_reverse(i, logN)] = x[i];
    }

    // Cooley-Tukey decimation-in-time radix-2 FFT
    for (int s = 1; s <= logN; s++)
    {
        int m = 1 << s;  // 2^s
        int mh = m >> 1; // mh - half m

        for (int i = 0; i < n / 2; i++)
        {
            int k = (i / mh) * m;
            int j = i % mh;
            int kj = k + j;

            float twr, twi;
            __sincosf(-M_PI * j / mh, &twi, &twr);

            complex32_t tw = complex32_t(twr, twi);

            complex32_t u = Y[kj];
            complex32_t v = Y[kj + mh] * tw;

            Y[kj] = u + v;
            Y[kj + mh] = u - v;
        }
    }
}

void fft_radix2_1024(complex32_t *x, complex32_t *Y)
{
    // this is the fft radix implementation for size of 1024
    constexpr int N = 1024;
    constexpr int logN = 10;

    constexpr int N4 = N / 4;

    for (int i = 0; i < N4; i += 4)
    {
        Y[bit_reverse(i, logN)] = x[i];
        Y[bit_reverse(i + 1, logN)] = x[i + 1];
        Y[bit_reverse(i + 2, logN)] = x[i + 2];
        Y[bit_reverse(i + 3, logN)] = x[i + 3];
    }

    for (int s = 1; s <= logN; s++)
    {
        int m = 1 << s;
        int mh = m >> 1;

        for (int i = 0; i < N4; i += 2)
        {
            int k = (i / mh) * m;
            int j = i % mh;
            int kj = k + j;

            float twr, twi;
            __sincosf(-M_PI * j / mh, &twi, &twr);

            complex32_t tw = complex32_t(twr, twi);

            complex32_t u = Y[kj];
            complex32_t v = Y[kj + mh] * tw;

            Y[kj] = u + v;
            Y[kj + mh] = u - v;

            // unrolling the loop

            k = (i + 1) / mh * m;
            j = (i + 1) % mh;

            kj = k + j;

            __sincosf(-M_PI * j / mh, &twi, &twr);

            tw = complex32_t(twr, twi);

            u = Y[kj];
            v = Y[kj + mh] * tw;

            Y[kj] = u + v;
            Y[kj + mh] = u - v;
        }
    }
}
