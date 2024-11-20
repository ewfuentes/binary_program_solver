project("Binary Program Solver" LANGUAGES CUDA CXX)
cmake_minimum_required(VERSION 3.22)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(GTest REQUIRED)

# Define a library
add_library(load_mps load_mps.cc)
set_target_properties(load_mps PROPERTIES
    PUBLIC_HEADER load_mps.hh)

# Define a test executable
add_executable(load_mps_test load_mps_test.cc)
target_link_libraries(load_mps_test GTest::gtest GTest::gtest_main load_mps)

# Defining a cuda executable is easy as long as it ends in .cu
add_executable(vector_add vector_add.cu)

