
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include "load_mps.hh"
#include "gpu_solver.hh"

class MPSGPUTest : public testing::TestWithParam<std::filesystem::path> {
};

TEST_P(MPSGPUTest, mps_file_test) {
    const auto &mps_data = load_mps_file(GetParam());

    solve_gpu(mps_data);
}

INSTANTIATE_TEST_SUITE_P(MPSGpuSolverTest, MPSGPUTest, testing::Values(
    "test_problems/stein9inf.mps",
    "test_problems/stein15inf.mps",
    "test_problems/stein45inf.mps"
));
