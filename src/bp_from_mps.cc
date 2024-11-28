
#include "bp_from_mps.hh"

#include <numeric>
#include <fmt/format.h>

template <typename T, typename A>
std::vector<std::string> sorted_keys(const T &seq, const A &accessor) {
    std::vector<std::string> out;
    out.reserve(seq.size());

    for (const auto &item : seq) {
        out.push_back(accessor(item));
    }

    std::sort(out.begin(), out.end());
    return out;
}


BinaryProgram bp_from_mps(const MPSData &mps) {
    using CT = RowInfo::Type;

    const int num_simple_constraints = 
        std::accumulate(mps.rows.begin(), mps.rows.end(), 0,
                [](const int accum, const auto &row){
                switch(row.type) {
                    case CT::NONE: return accum;
                    case CT::LESS_THAN: 
                    case CT::GREATER_THAN: return accum + 1;
                    case CT::EQUAL: return accum + 2;
                    default: return accum;
                };
            });

    BinaryProgram out = {
        .objective = Eigen::VectorXf(mps.columns.size()),
        .constraints = BinaryProgram::ConstraintMat(num_simple_constraints, mps.columns.size()),
        .rhs = Eigen::VectorXf(num_simple_constraints)
    };

    const std::vector<std::string> variables =
        sorted_keys(mps.columns, [](const auto &item){ return item.first; });

    const std::vector<std::string> constraints =
        sorted_keys(mps.rows, [](const auto &item){ return item.name; });

    std::unordered_map<std::string, CT> constraint_type_from_name;
    std::transform(mps.rows.begin(), mps.rows.end(), 
                   std::inserter(constraint_type_from_name, constraint_type_from_name.end()),
                   [](const auto &row){ return std::pair{row.name, row.type}; });

    std::unordered_map<std::string, int> constraint_start_idx_from_name;
    {
        int start_idx = 0;
        for (const auto &constraint_name : constraints) {
            const auto constraint_type = constraint_type_from_name.at(constraint_name);
            if (constraint_type == CT::NONE || constraint_type == CT::UNKNOWN) {
                continue;
            }
            constraint_start_idx_from_name[constraint_name] = start_idx;

            start_idx += constraint_type == CT::EQUAL ? 2 : 1;
        }
    }

    // fmt::print("variables: {}\r\n", variables);
    // fmt::print("constraints: {}\r\n", constraints);
    // fmt::print("constraint_type_from_name: {}\r\n", constraint_type_from_name);
    // fmt::print("constraint_start_idx_from_name: {}\r\n", constraint_start_idx_from_name);

    std::vector<Eigen::Triplet<float>> constraint_entries;
    for (int var_idx = 0; var_idx < static_cast<int>(variables.size()); var_idx++) {
        const std::string &var_name = variables.at(var_idx);
        const auto &entries = mps.columns.at(var_name);
        // fmt::print("Entries for {}: {}\r\n", var_name, entries);
        for (const auto &[constraint_name, value] : entries) {
            const auto true_constraint_type = constraint_type_from_name.at(constraint_name);

            if (true_constraint_type == CT::NONE) {
                out.objective(var_idx) = value;
                continue;
            } else if (true_constraint_type == CT::UNKNOWN) {
                continue;
            }

            const auto constraint_types = true_constraint_type == CT::EQUAL ? 
                std::vector<CT>{CT::LESS_THAN, CT::GREATER_THAN} 
                : std::vector<CT>{true_constraint_type};
            const int constraint_start_idx = constraint_start_idx_from_name.at(constraint_name);

            for (int i = 0; i < static_cast<int>(constraint_types.size()); i++) {
                const auto constraint_type = constraint_types.at(i);
                constraint_entries.push_back(
                    {constraint_start_idx + i, var_idx, 
                        constraint_type == CT::LESS_THAN ? value : -value});
            }
        }
    }

    for (int constraint_idx = 0; constraint_idx < static_cast<int>(constraints.size()); constraint_idx++) {
        const std::string &constraint_name = constraints.at(constraint_idx);
        const auto true_constraint_type = constraint_type_from_name.at(constraint_name);
        if (true_constraint_type == CT::NONE || true_constraint_type == CT::UNKNOWN) {
            continue;
        }
        const auto rhs_iter = mps.rhs.find(constraint_name);
        float rhs_value = 0.0;
        if (rhs_iter != mps.rhs.end()) {
            rhs_value = rhs_iter->second;
        } else {
            fmt::print("Cannot find constraint \"{}\" in RHS. Assuming it's equal to zero\r\n", constraint_name);
        }
        const int constraint_start_idx = constraint_start_idx_from_name.at(constraint_name);
        const auto constraint_types = true_constraint_type == CT::EQUAL ? 
            std::vector<CT>{CT::LESS_THAN, CT::GREATER_THAN} 
            : std::vector<CT>{true_constraint_type};
        for (int i = 0; i < static_cast<int>(constraint_types.size()); i++) {
            const auto constraint_type = constraint_types.at(i);
            out.rhs(constraint_start_idx + i) = constraint_type == CT::LESS_THAN ? rhs_value : -rhs_value;
        }
    }

    out.constraints.setFromTriplets(constraint_entries.begin(), constraint_entries.end());

    return out;
}
