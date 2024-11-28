
#include "cpu_solver.hh"
#include "load_mps.hh"
#include "bp_from_mps.hh"

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

namespace {
BinaryProgram generate_simple_binary_program() {
    Eigen::Vector4f objective{1.0f, 2.0f, 3.0f, 4.0f};
    Eigen::Vector4f rhs{-1.0, -1.0, 1.0, 1.0};
    Eigen::SparseMatrix<float, Eigen::RowMajor> constraints(4, 4);
    const std::vector<Eigen::Triplet<float>> constraint_entries = {
        {0, 0, -1.0f},
        {0, 1, -1.0f},
        {1, 2, -1.0f},
        {1, 3, -1.0f},
        {2, 0, 1.0f},
        {2, 1, 1.0f},
        {3, 2, 1.0f},
        {3, 3, 1.0f},
    };
    constraints.setFromTriplets(constraint_entries.begin(), constraint_entries.end());
    return BinaryProgram{
        .objective = objective,
        .constraints = constraints,
        .rhs = rhs
    };
}
}

TEST(CpuSolverTest, test_simple_program) {
    const auto program = generate_simple_binary_program();
    const auto result = solve_cpu(program);

    constexpr double TOL = 1e-6;
    EXPECT_NEAR(result.value, 4.0, TOL);
    EXPECT_EQ(result.assignment, Eigen::Vector4f(1.0, 0.0, 1.0, 0.0));

    std::cout << "Assignment: " << result.assignment.transpose() << std::endl;
    std::cout << "Value: " << result.value << std::endl;
}

class MPSCPUTest : public testing::TestWithParam<std::filesystem::path> {
};

TEST_P(MPSCPUTest, mps_file_test) {
    const auto mps_data = load_mps_file(std::filesystem::canonical("/proc/self/exe").parent_path() / GetParam());
    const auto binary_program = bp_from_mps(mps_data);
    const auto result = solve_cpu(binary_program);

    std::cout << "Assignment: " << result.assignment.transpose() << std::endl;
    std::cout << "Value: " << result.value << std::endl;
}

INSTANTIATE_TEST_SUITE_P(MPSTests, MPSCPUTest, testing::Values(
    "test_problems/stein9inf.mps",
    "test_problems/stein15inf.mps"
//    "test_problems/stein45inf.mps",
//    "test_problems/p0201.mps",
//    "test_problems/p2m2p1m1p0n100.mps",
//    "test_problems/glass-sc.mps"
));

