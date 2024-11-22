
#include "cpu_solver.hh"
#include "binary_program.hh"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {
using BP = BinaryProgram;
}

namespace detail {

bool check_if_feasible(
    const BP::ConstraintMat &A, const Eigen::VectorXf &rhs, const PartialAssignment &x) {
    const int num_assigned = x.rows();
    const int num_unassigned = A.cols() - num_assigned;
    Eigen::VectorXf optimistic_constraints = Eigen::VectorXf(A.rows());

    for (int row_idx = 0; row_idx < A.rows(); row_idx++) {
        float constraint_value = 0;
        for (BP::ConstraintMat::InnerIterator it(A, row_idx); it; ++it) {
            if (it.col() < num_assigned) {
                constraint_value += x(it.col()) * it.value();
            } else {
                constraint_value += std::min(it.value(), 0.0f);
            }
        }
        optimistic_constraints(row_idx) = constraint_value;
    }

    const Eigen::ArrayX<bool> are_constraints_satisfied =
        optimistic_constraints.array() <= rhs.array();
    return are_constraints_satisfied.all();
}

float compute_optimistic_cost(const Eigen::VectorXf &obj, const PartialAssignment &x) {
    const int num_assigned = x.rows();
    const int num_unassigned = obj.rows() - num_assigned;
    const float realized_cost = obj.topRows(num_assigned).transpose() * x;
    const float optimistic_cost_to_go = obj.bottomRows(num_unassigned).cwiseMin(0).sum();
    return realized_cost + optimistic_cost_to_go;
}

double logsumexp(const double a, const double b) {
    const double c = std::max(a, b);
    return c + std::log2(std::exp2(a - c) + std::exp2(b - c));
}
} // namespace detail


Solution solve_cpu(const BinaryProgram &program) {
    const int num_variables = program.objective.rows();

    std::vector<PartialAssignment> queue{PartialAssignment()};

    PartialAssignment best_solution;
    float best_cost = std::numeric_limits<float>::max();

    int pop_count = 0;
    int early_term = 0;
    double log_coverage = std::numeric_limits<double>::lowest();
    while (!queue.empty()) {
        const PartialAssignment work = queue.back();
        queue.pop_back();
        pop_count++;

        if (pop_count % 1000 == 0) {
            std::cout << "pop_count: " << pop_count
                << " early term count: " << early_term
                << " log coverage: " << log_coverage  << " / " << num_variables
                << std::endl;
        }

        // Check if feasible
        const bool is_feasible = detail::check_if_feasible(program.constraints, program.rhs, work);
        const float optimistic_cost = detail::compute_optimistic_cost(program.objective, work);

        if (!is_feasible || optimistic_cost >= best_cost) {
            early_term++;
            log_coverage = detail::logsumexp(log_coverage, num_variables - work.size());
            continue;
        }

        if (work.rows() == program.objective.rows() && optimistic_cost < best_cost) {
            // This is a complete assigment, store if better than incumbent
            best_cost = optimistic_cost;
            best_solution = work;
            log_coverage = detail::logsumexp(log_coverage, num_variables - work.size());
            std::cout << "Found new best incumbent cost: " << best_cost <<std::endl;
            continue;
        }

        {
            PartialAssignment assign_true(work.rows() + 1);
            assign_true.topRows(work.rows()) = work;
            assign_true(work.rows()) = 1.0f;
            queue.push_back(assign_true);
        }
        {
            PartialAssignment assign_false(work.rows() + 1);
            assign_false.topRows(work.rows()) = work;
            assign_false(work.rows()) = 0.0f;
            queue.push_back(assign_false);
        }
    }

    return {
        .assignment = best_solution,
        .value = best_cost,
    };
}
