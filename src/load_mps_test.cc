#include <fmt/base.h>
#include <gtest/gtest.h>

#include "cpu_solver.hh"
#include "load_mps.hh"
#include "fmt/format.h"
#include "fmt/ranges.h"

namespace {
MPSData create_test_mps_data() {
    // Define a test program:
    // x = [x1, x2, x3, x4]
    // min c x
    // s. t. 1 x1 + 2 x2 + 3 x3 + 4 x4 >= 1 # constraint_1
    //       5 x1        + 6 x3        <= 3 # constraint_2
    //              7 x2 + 8 x3         = 5 # constraint_3
    //                            9 x4 <= 7 # constraint_3
    //
    // c = [10, 20, 30, 40]

    return {
        .name = "test_program",
        .rows = {
            {.name = "obj", .type = RowInfo::Type::NONE},
            {.name = "constraint_1", .type = RowInfo::Type::GREATER_THAN},
            {.name = "constraint_2", .type = RowInfo::Type::LESS_THAN},
            {.name = "constraint_3", .type = RowInfo::Type::EQUAL},
            {.name = "constraint_4", .type = RowInfo::Type::LESS_THAN},
        },
        .columns = {
            {"x1", {
                {"obj", 10},
                {"constraint_1", 1},
                {"constraint_2", 5}}},
            {"x2", {
                {"obj", 20},
                {"constraint_1", 2},
                {"constraint_3", 7}}},
            {"x3", {
                {"obj", 30},
                {"constraint_1", 3},
                {"constraint_2", 6},
                {"constraint_3", 8}}},
            {"x4", {
                {"obj", 40},
                {"constraint_1", 4},
                {"constraint_4", 9}}},
        },
        .rhs = {
            {"constraint_1", 1},
            {"constraint_2", 3},
            {"constraint_3", 5},
            {"constraint_4", 7},
        }
    };
}
}

std::string to_string(const RowInfo::Type &t) {
    switch(t) {
        case RowInfo::Type::NONE:
            return "None";
        case RowInfo::Type::LESS_THAN:
            return "Less Than";
        case RowInfo::Type::GREATER_THAN:
            return "Greater Than";
        case RowInfo::Type::EQUAL:
            return "Equal";
        default:
            return "Unknown";
    }
}

template <>
struct fmt::formatter<RowInfo::Type>  {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

    auto format(const RowInfo::Type &t, format_context& ctx) const -> format_context::iterator {
      string_view name = "unknown";
      switch (t) {
        case RowInfo::Type::NONE:   name = "None"; break;
        case RowInfo::Type::LESS_THAN:   name = "LE"; break;
        case RowInfo::Type::GREATER_THAN:   name = "GE"; break;
        case RowInfo::Type::EQUAL:   name = "EQ"; break;
        default: name = "UNKNOWN"; break;
      }
      return fmt::format_to(ctx.out(), "{}", name);
    }
};

template <>
struct fmt::formatter<RowInfo> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

    auto format(const RowInfo &c, format_context& ctx) const {
      return format_to(ctx.out(), "{{Row Info name: {} type: {}}}", c.name, c.type);
    }
};

TEST(LoadMpsTest, load_simple_example) {
    MPSData mps = load_mps_file(std::filesystem::canonical("/proc/self/exe").parent_path() / "test_problems/sample.mps");

    EXPECT_EQ(mps.columns.size(), 10);
    EXPECT_EQ(mps.rows.size(), 3);
}

TEST(LoadMpsTest, load_dual_columns) {
    MPSData mps = load_mps_file(std::filesystem::canonical("/proc/self/exe").parent_path() / "test_problems/stein9inf.mps");

    EXPECT_EQ(mps.columns.size(), 9);
    EXPECT_EQ(mps.rows.size(), 15);
}


TEST(LoadMpsTest, load_full_example) {
    MPSData mps = load_mps_file(std::filesystem::canonical("/proc/self/exe").parent_path() / "test_problems/glass-sc.mps");

    EXPECT_EQ(mps.columns.size(), 214);
    EXPECT_EQ(mps.rows.size(), 6120);
}

TEST(BPFromMPSTest, test_case) {
    const MPSData test_mps_data = create_test_mps_data();

    const BinaryProgram bp = bp_from_mps(test_mps_data);

    std::cout << "obj: " << bp.objective.transpose() << std::endl;
    std::cout << "rhs: " << bp.rhs.transpose() << std::endl;
    std::cout << "constraints: " << std::endl << bp.constraints << std::endl;

    constexpr double TOL = 1e-6;
    // Check the objective
    const Eigen::VectorXf expected_objective = (Eigen::VectorXf(4) << 10, 20, 30, 40).finished();
    ASSERT_EQ(bp.objective.rows(), expected_objective.rows());
    EXPECT_NEAR((bp.objective - expected_objective).norm(), 0.0, TOL);

    // Check the RHS
    const Eigen::VectorXf expected_rhs = (Eigen::VectorXf(5) << -1, 3, 5, -5, 7).finished();
    ASSERT_EQ(bp.rhs.rows(), expected_rhs.rows());
    EXPECT_NEAR((bp.rhs - expected_rhs).norm(), 0.0, TOL);

    // Check the constraint matrix
    const Eigen::MatrixXf expected_constraints = 
        (Eigen::MatrixXf(5, 4) << -1, -2, -3, -4,
                                   5,  0,  6,  0,
                                   0,  7,  8,  0,
                                   0, -7, -8,  0,
                                   0,  0,  0,  9).finished();
    ASSERT_EQ(bp.constraints.rows(), expected_constraints.rows());
    ASSERT_EQ(bp.constraints.cols(), expected_constraints.cols());
    EXPECT_NEAR((bp.constraints -  expected_constraints).norm(), 0.0, TOL);
}
