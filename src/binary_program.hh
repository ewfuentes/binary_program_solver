
#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

struct BinaryProgram {
    using ConstraintMat = Eigen::SparseMatrix<float, Eigen::RowMajor>;
    // An optimization problem of the form:
    // min objective.transpose() * x
    // s.t. A * x <= rhs
    //      x \in {0, 1} ^ N
    Eigen::VectorXf objective;
    ConstraintMat constraints;
    Eigen::VectorXf rhs;
};

struct Solution {
    Eigen::VectorXf assignment;
    float value;
};
