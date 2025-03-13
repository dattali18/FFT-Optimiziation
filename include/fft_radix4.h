#pragma once

#include <cstdint>
#include <cmath>

#include "complex.h"

void fft_radix4(complex32_t *x, complex32_t *y, int n);

void fft_radix4_1024_v1(complex32_t *x, complex32_t *y);
