#pragma once

#include "complex.h"

void butterfly_radix2(complex32_t data[2]);

void butterfly_radix4(complex32_t data[4]);

void butterfly_radix8(complex32_t data[8]);

void butterfly_radix16(complex32_t data[16]);

void fft_radix2_1024(complex32_t *x, complex32_t *Y);

void fft_radix4_1024(complex32_t *x, complex32_t *Y);

void fft_radix8_1024(complex32_t *x, complex32_t *Y);

void fft_radix16_1024(complex32_t *x, complex32_t *Y);

void init_twiddle_table();

static complex32_t tw_table[1024];
