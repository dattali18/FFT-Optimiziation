cmake_minimum_required(VERSION 3.10)

# Set the project name
project(fft_optimization)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add include directories
include_directories(include)
# my include dir for this project is include/
set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add the executable
file(GLOB SOURCES "src/*.cpp")
add_executable(fft_optimization ${SOURCES})