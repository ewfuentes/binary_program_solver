
#include "cpu_solver.hh"
#include "binary_program.hh"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

namespace {
using BP = BinaryProgram;
}

namespace detail {

bool check_if_feasible(
    const BP::ConstraintMat &A, const Eigen::VectorXf &rhs, const PartialAssignment &x) {
    const int num_assigned = x.num_assigned;
    const int num_unassigned = A.cols() - num_assigned;
    Eigen::VectorXf optimistic_constraints = Eigen::VectorXf(A.rows());

    for (int row_idx = 0; row_idx < A.rows(); row_idx++) {
        float constraint_value = 0;
        for (BP::ConstraintMat::InnerIterator it(A, row_idx); it; ++it) {
            if (it.col() < num_assigned) {
                constraint_value += x.assignment(it.col()) * it.value();
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
    const int num_assigned = x.num_assigned;
    const int num_unassigned = obj.rows() - num_assigned;
    const float realized_cost = obj.topRows(num_assigned).transpose() * x.assignment.topRows(x.num_assigned);
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

    std::vector<std::vector<float>> queue{{1, 0}};

    Eigen::VectorXf best_solution;
    float best_cost = std::numeric_limits<float>::max();

    int pop_count = 0;
    int early_term = 0;
    double log_coverage = std::numeric_limits<double>::lowest();
    PartialAssignment current = {
        .assignment = Eigen::VectorXf(num_variables),
        .num_assigned = 1,
    };
    while (!queue.empty()) {
        // Update the partial assignment
        const int curr_depth = queue.size();
        for (int i = 0; i < queue.size(); i++) {
            current.assignment(i) = queue.at(i).back();
        }
        current.num_assigned = curr_depth;
        pop_count++;

        if (pop_count % 10000 == 0) {
            fmt::print("pop count: {} early_term count: {} best_cost: {} log_coverage: {}\r\n",
                       pop_count, early_term, best_cost, log_coverage - num_variables);
        }

        // Check if feasible
        const bool is_feasible = detail::check_if_feasible(program.constraints, program.rhs, current);
        const float optimistic_cost = detail::compute_optimistic_cost(program.objective, current);
    
        if (!is_feasible || optimistic_cost >= best_cost) {
            //This is a dead branch, remove the current assignment from the current domain
            early_term++;
            log_coverage = detail::logsumexp(log_coverage, num_variables - curr_depth);
            queue.back().pop_back();
        } else if (curr_depth == num_variables && optimistic_cost < best_cost) {
            // This is a complete assigment, store if better than incumbent
            best_cost = optimistic_cost;
            best_solution = current.assignment;
            log_coverage = detail::logsumexp(log_coverage, num_variables - curr_depth);
            fmt::print("Found new best incumbent cost: {}\r\n", optimistic_cost);
            queue.back().pop_back();
        } else {
            // We haven't yet assigned all variables, but we also haven't found a dead branch
            // add the domain for the next variable to the queue
            queue.push_back({1, 0});
        }

        while (queue.back().empty()) {
            // We've exhausted all possible assignments for the current variable
            // remove the frame from the current variable, and the last item of the
            // previous variable. Repeat this process if the last variable is now empty
            queue.pop_back();
            if (!queue.empty()) {
                queue.back().pop_back();
            }
        } 
    }

    return {
        .assignment = best_solution,
        .value = best_cost,
    };
}
