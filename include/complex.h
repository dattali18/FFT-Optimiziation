#pragma once

#include <complex>

/// This file will define a complex type

typedef std::complex<float> complex32_t;
typedef std::complex<double> complex64_t;

constexpr complex32_t I = complex32_t(0.0f, 1.0f);