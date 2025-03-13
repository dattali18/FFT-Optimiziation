#include "fft.h"

void butterfly_radix2(complex32_t data[2])
{
    complex32_t temp = data[0];
    data[0] = temp + data[1];
    data[1] = temp - data[1];
}

void butterfly_radix4(complex32_t data[4])
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

void butterfly_radix8(complex32_t data[8])
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

void butterfly_radix16(complex32_t data[16])
{
    // First level: 8 radix-2 butterflies
    complex32_t temp[16];
    for (int j = 0; j < 8; j++)
    {
        temp[j] = data[j] + data[j + 8];
        temp[j + 8] = data[j] - data[j + 8];
    }

    // Second level: 4 radix-4 butterflies
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

void fft_radix2_1024(complex32_t *x, complex32_t *Y)
{
    extern complex32_t tw_table[1024];

    for (int i = 0; i < 1024; i++)
    {
        Y[i] = x[i];
    }

    for (int s = 0; s < 10; s++)
    {
        int mh = 1 << s; // 1, 2, 4, 8, ...
        int m = mh * 2;  // 2, 4, 8, 16, ...

        // Single loop from 0 to N/2 (512)
        for (int i = 0; i < 512; i++)
        {
            int jk = (i & (~(mh - 1))) | (i & (mh - 1));
            int jkmh = jk + mh;

            complex32_t data[2];
            data[0] = Y[jk];
            data[1] = Y[jkmh] * tw_table[(jk % m) * 1024 / m]; // Calculate twiddle index

            butterfly_radix2(data);

            Y[jk] = data[0];
            Y[jkmh] = data[1];
        }
    }
}

void fft_radix4_1024(complex32_t *x, complex32_t *Y)
{
    extern complex32_t tw_table[1024]; // Assuming tw_table is defined

    for (int i = 0; i < 1024; i++)
    {
        Y[i] = x[i];
    }

    for (int s = 0; s < 5; s++)
    {                          // 1024 = 4^5, so 5 stages
        int mh = 1 << (2 * s); // 1, 4, 16, 64, 256
        int m = mh * 4;        // 4, 16, 64, 256, 1024

        // Single loop from 0 to N/4 (256)
        for (int i = 0; i < 256; i++)
        {
            int jk = (i & (~(mh - 1))) | (i & (mh - 1));
            int jkmh = jk + mh;
            int jk2mh = jk + 2 * mh;
            int jk3mh = jk + 3 * mh;

            complex32_t data[4];
            data[0] = Y[jk];
            data[1] = Y[jkmh] * tw_table[(jk % m) * 1024 / m];
            data[2] = Y[jk2mh] * tw_table[(2 * jk % m) * 1024 / m];
            data[3] = Y[jk3mh] * tw_table[(3 * jk % m) * 1024 / m];

            butterfly_radix4(data);

            Y[jk] = data[0];
            Y[jkmh] = data[1];
            Y[jk2mh] = data[2];
            Y[jk3mh] = data[3];
        }
    }
}

void fft_radix8_1024(complex32_t *x, complex32_t *Y)
{
    extern complex32_t tw_table[1024]; // Assuming tw_table is defined

    for (int i = 0; i < 1024; i++)
    {
        Y[i] = x[i];
    }

    // Radix-8 Stage (1024/8 = 128)
    int mh8 = 1;
    int m8 = 8;
    for (int i = 0; i < 128; i++)
    {
        int jk = (i & (~(mh8 - 1))) | (i & (mh8 - 1));
        int jkmh = jk + mh8;
        int jk2mh = jk + 2 * mh8;
        int jk3mh = jk + 3 * mh8;
        int jk4mh = jk + 4 * mh8;
        int jk5mh = jk + 5 * mh8;
        int jk6mh = jk + 6 * mh8;
        int jk7mh = jk + 7 * mh8;

        complex32_t data[8];
        data[0] = Y[jk];
        data[1] = Y[jkmh] * tw_table[(jk % m8) * 1024 / m8];
        data[2] = Y[jk2mh] * tw_table[(2 * jk % m8) * 1024 / m8];
        data[3] = Y[jk3mh] * tw_table[(3 * jk % m8) * 1024 / m8];
        data[4] = Y[jk4mh] * tw_table[(4 * jk % m8) * 1024 / m8];
        data[5] = Y[jk5mh] * tw_table[(5 * jk % m8) * 1024 / m8];
        data[6] = Y[jk6mh] * tw_table[(6 * jk % m8) * 1024 / m8];
        data[7] = Y[jk7mh] * tw_table[(7 * jk % m8) * 1024 / m8];

        butterfly_radix8(data);

        Y[jk] = data[0];
        Y[jkmh] = data[1];
        Y[jk2mh] = data[2];
        Y[jk3mh] = data[3];
        Y[jk4mh] = data[4];
        Y[jk5mh] = data[5];
        Y[jk6mh] = data[6];
        Y[jk7mh] = data[7];
    }

    // Radix-2 Stages (1024/2 = 512, 9 more stages)
    for (int s = 0; s < 9; s++)
    {
        int mh2 = 1 << s;
        int m2 = mh2 * 2;

        for (int i = 0; i < 512; i++)
        {
            int jk = (i & (~(mh2 - 1))) | (i & (mh2 - 1));
            int jkmh = jk + mh2;

            complex32_t data[2];
            data[0] = Y[jk];
            data[1] = Y[jkmh] * tw_table[(jk % m2) * 1024 / m2];

            butterfly_radix2(data);

            Y[jk] = data[0];
            Y[jkmh] = data[1];
        }
    }
}

void fft_radix16_1024(complex32_t *x, complex32_t *Y)
{
    extern complex32_t tw_table[1024];

    for (int i = 0; i < 1024; i++)
    {
        Y[i] = x[i];
    }

    // Radix-16 Stage (1024/16 = 64)
    int mh16 = 1;
    int m16 = 16;
    for (int i = 0; i < 64; i++)
    {
        // ... (radix-16 butterfly code, same as before)
        int jk = (i & (~(mh16 - 1))) | (i & (mh16 - 1));
        int jkmh = jk + mh16;
        int jk2mh = jk + 2 * mh16;
        int jk3mh = jk + 3 * mh16;
        int jk4mh = jk + 4 * mh16;
        int jk5mh = jk + 5 * mh16;
        int jk6mh = jk + 6 * mh16;
        int jk7mh = jk + 7 * mh16;
        int jk8mh = jk + 8 * mh16;
        int jk9mh = jk + 9 * mh16;
        int jk10mh = jk + 10 * mh16;
        int jk11mh = jk + 11 * mh16;
        int jk12mh = jk + 12 * mh16;
        int jk13mh = jk + 13 * mh16;
        int jk14mh = jk + 14 * mh16;
        int jk15mh = jk + 15 * mh16;

        complex32_t data[16];
        data[0] = Y[jk];
        data[1] = Y[jkmh] * tw_table[(jk % m16) * 1024 / m16];
        data[2] = Y[jk2mh] * tw_table[(2 * jk % m16) * 1024 / m16];
        data[3] = Y[jk3mh] * tw_table[(3 * jk % m16) * 1024 / m16];
        data[4] = Y[jk4mh] * tw_table[(4 * jk % m16) * 1024 / m16];
        data[5] = Y[jk5mh] * tw_table[(5 * jk % m16) * 1024 / m16];
        data[6] = Y[jk6mh] * tw_table[(6 * jk % m16) * 1024 / m16];
        data[7] = Y[jk7mh] * tw_table[(7 * jk % m16) * 1024 / m16];
        data[8] = Y[jk8mh] * tw_table[(8 * jk % m16) * 1024 / m16];
        data[9] = Y[jk9mh] * tw_table[(9 * jk % m16) * 1024 / m16];
        data[10] = Y[jk10mh] * tw_table[(10 * jk % m16) * 1024 / m16];
        data[11] = Y[jk11mh] * tw_table[(11 * jk % m16) * 1024 / m16];
        data[12] = Y[jk12mh] * tw_table[(12 * jk % m16) * 1024 / m16];
        data[13] = Y[jk13mh] * tw_table[(13 * jk % m16) * 1024 / m16];
        data[14] = Y[jk14mh] * tw_table[(14 * jk % m16) * 1024 / m16];
        data[15] = Y[jk15mh] * tw_table[(15 * jk % m16) * 1024 / m16];

        butterfly_radix16(data);
        // ... (write back to Y)
        Y[jk] = data[0];
        Y[jkmh] = data[1];
        Y[jk2mh] = data[2];
        Y[jk3mh] = data[3];
        Y[jk4mh] = data[4];
        Y[jk5mh] = data[5];
        Y[jk6mh] = data[6];
        Y[jk7mh] = data[7];
        Y[jk8mh] = data[8];
        Y[jk9mh] = data[9];
        Y[jk10mh] = data[10];
        Y[jk11mh] = data[11];
        Y[jk12mh] = data[12];
        Y[jk13mh] = data[13];
        Y[jk14mh] = data[14];
        Y[jk15mh] = data[15];
    }

    // Radix-8 Stage (1024/16/8 = 8)
    int mh8 = 16;
    int m8 = 128;
    for (int i = 0; i < 8; i++)
    {
        // ... (radix-8 butterfly code)
        int jk = (i & (~(mh8 - 1))) | (i & (mh8 - 1));
        int jkmh = jk + mh8;
        int jk2mh = jk + 2 * mh8;
        int jk3mh = jk + 3 * mh8;
        int jk4mh = jk + 4 * mh8;
        int jk5mh = jk + 5 * mh8;
        int jk6mh = jk + 6 * mh8;
        int jk7mh = jk + 7 * mh8;

        complex32_t data[8];
        data[0] = Y[jk];
        data[1] = Y[jkmh] * tw_table[(jk % m8) * 1024 / m8];
        data[2] = Y[jk2mh] * tw_table[(2 * jk % m8) * 1024 / m8];
        data[3] = Y[jk3mh] * tw_table[(3 * jk % m8) * 1024 / m8];
        data[4] = Y[jk4mh] * tw_table[(4 * jk % m8) * 1024 / m8];
        data[5] = Y[jk5mh] * tw_table[(5 * jk % m8) * 1024 / m8];
        data[6] = Y[jk6mh] * tw_table[(6 * jk % m8) * 1024 / m8];
        data[7] = Y[jk7mh] * tw_table[(7 * jk % m8) * 1024 / m8];

        butterfly_radix8(data);
        // ... (write back to Y)
        Y[jk] = data[0];
        Y[jkmh] = data[1];
        Y[jk2mh] = data[2];
        Y[jk3mh] = data[3];
        Y[jk4mh] = data[4];
        Y[jk5mh] = data[5];
        Y[jk6mh] = data[6];
        Y[jk7mh] = data[7];
    }

    // Radix-2 Stages (1024/16/8/2 = 4, 2 more stages)
    for (int s = 0; s < 2; s++)
    {
        int mh2 = 64 << s; // 64, 128
        int m2 = mh2 * 2;  // 128, 256

        for (int i = 0; i < 512; i++)
        {
            int jk = (i & (~(mh2 - 1))) | (i & (mh2 - 1));
            int jkmh = jk + mh2;

            complex32_t data[2];
            data[0] = Y[jk];
            data[1] = Y[jkmh] * tw_table[(jk % m2) * 1024 / m2];

            butterfly_radix2(data);

            Y[jk] = data[0];
            Y[jkmh] = data[1];
        }
    }
}

void init_twiddle_table()
{
    for (int i = 0; i < 1024; i++)
    {
        float twr, twi;
        __sincosf(-2.0f * M_PI * i / 1024, &twi, &twr);
        tw_table[i] = complex32_t(twr, twi);
    }
}