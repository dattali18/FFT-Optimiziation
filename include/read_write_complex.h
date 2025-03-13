#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "complex.h"

// function to read a complex number from a file given a file name
void read_complex_from_file(const std::string& filename, complex32_t* arr, int size);

void write_complex_from_file(const std::string& filename, complex32_t* arr, int size);