#pragma once

#include <cstdint>
#include <cmath>

#include "complex.h"

void fft_radix2(complex32_t *x, complex32_t *Y, int n);

void fft_radix2_1024_v1(complex32_t *x, complex32_t *Y);
