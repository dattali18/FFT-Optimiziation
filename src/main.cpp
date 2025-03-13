#include <iostream>

#include "fft.h"

#include "read_write_complex.h"
#include "complex.h"

#include "cpu_timer.h"
#include "helper.h"

// this program takes two arguments the first is the input filename and the second is the output filename
int main(int argc, char *argv[])
{
    // check if the number of arguments is correct
    // if (argc != 4)
    // {
    //     std::cerr << "Usage: " << argv[0] << " <length> <input_file> <output_file>" << std::endl;
    //     return 1;
    // }

    // int length = std::stoi(argv[1]);
    // std::string input_filename = argv[2];
    // std::string output_filename = argv[3];

    constexpr int length = 1024;

    complex32_t *input_data = new complex32_t[length];
    complex32_t *output_data = new complex32_t[length];

    // read the input data
    // read_complex_from_file(input_filename, input_data, length);

#define PROFILE

#ifdef PROFILE
    MIN_TIME(fft_radix2_1024(input_data, output_data));
    MIN_TIME(fft_radix4_1024(input_data, output_data));
    MIN_TIME(fft_radix8_1024(input_data, output_data));
    MIN_TIME(fft_radix16_1024(input_data, output_data));
#else
    fft_radix4(input_data, output_data, length);
#endif

    // write the output data
    // write_complex_from_file(output_filename, output_data, length);

    delete[] input_data;
    delete[] output_data;

    return 0;
}