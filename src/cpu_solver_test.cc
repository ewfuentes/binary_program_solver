
#include "cpu_solver.hh"

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
