project("Binary Program Solver" LANGUAGES CUDA CXX)
cmake_minimum_required(VERSION 3.22)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_BUILD_TYPE Release)

find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)

# Add Binary Program 
add_library(binary_program INTERFACE)
target_link_libraries(binary_program INTERFACE Eigen3::Eigen)
set_target_properties(binary_program PROPERTIES
    PUBLIC_HEADER binary_program.hh)

# Define a library
add_library(load_mps load_mps.cc)
set_target_properties(load_mps PROPERTIES
    PUBLIC_HEADER load_mps.hh)
target_link_libraries(load_mps binary_program fmt::fmt)

# Define a test executable
add_executable(load_mps_test load_mps_test.cc)
target_link_libraries(load_mps_test PRIVATE GTest::gtest GTest::gtest_main load_mps)
configure_file(test_problems/sample.mps test_problems/sample.mps)
configure_file(test_problems/stein9inf.mps test_problems/stein9inf.mps)
configure_file(test_problems/stein15inf.mps test_problems/stein15inf.mps)
configure_file(test_problems/stein45inf.mps test_problems/stein45inf.mps)
configure_file(test_problems/p0201.mps test_problems/p0201.mps)
configure_file(test_problems/p2m2p1m1p0n100.mps test_problems/p2m2p1m1p0n100.mps)
configure_file(test_problems/glass-sc.mps test_problems/glass-sc.mps)


# Add CPU solver
add_library(cpu_solver cpu_solver.cc)
set_target_properties(cpu_solver PROPERTIES
    PUBLIC_HEADER cpu_solver.hh)
target_link_libraries(cpu_solver PUBLIC binary_program load_mps)

# Add CPU Solver Test
add_executable(cpu_solver_test cpu_solver_test.cc)
target_link_libraries(cpu_solver_test PRIVATE GTest::gtest GTest::gtest_main cpu_solver)

# Defining a cuda executable is easy as long as it ends in .cu
add_executable(vector_add vector_add.cu)

