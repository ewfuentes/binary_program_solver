
#pragma once

#include "binary_program.hh"

using PartialAssignment = Eigen::VectorXf;

Solution solve_cpu(const BinaryProgram &program);

namespace detail {
bool is_feasible(const BinaryProgram::ConstraintMat &A, const Eigen::VectorXf &rhs, const PartialAssignment &x);
bool optimistic_cost(const Eigen::VectorXf &obj, const PartialAssignment &x);
}
