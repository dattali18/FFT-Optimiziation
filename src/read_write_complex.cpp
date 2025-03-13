#include "read_write_complex.h"

void read_complex_from_file(const std::string &filename, complex32_t *arr, int size)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error: could not open file " << filename << std::endl;
        return;
    }

    for (int i = 0; i < size; ++i)
    {
        float real, imag;
        file >> real >> imag;
        arr[i] = complex32_t(real, imag);
    }

    file.close();
}

void write_complex_from_file(const std::string &filename, complex32_t *arr, int size)
{
    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error: could not open file " << filename << std::endl;
        return;
    }

    for (int i = 0; i < size; ++i)
    {
        file << arr[i].real() << " " << arr[i].imag() << std::endl;
    }

    file.close();
}
