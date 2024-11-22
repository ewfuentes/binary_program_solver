
#include "load_mps.hh"
#include "binary_program.hh"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <iostream>
#include <numeric>
#include <string>
#include <fmt/format.h>
#include <fmt/ranges.h>

namespace {
void trim(std::string &s) {
    s.erase(0, s.find_first_not_of(" \t\r\n"));
    s.erase(s.find_last_not_of(" \t\r\n") + 1);
}

RowInfo::Type convertRowType(const std::string &rowTypeStr) {
    if (rowTypeStr == "N") {
        return RowInfo::Type::NONE;
    } else if (rowTypeStr == "L") {
        return RowInfo::Type::LESS_THAN;
    } else if (rowTypeStr == "G") {
        return RowInfo::Type::GREATER_THAN;
    } else if (rowTypeStr == "E") {
        return RowInfo::Type::EQUAL;
    } else {
        return RowInfo::Type::UNKNOWN; // For unsupported or invalid row types
    }
}
}

// Function to parse an MPS file
MPSData load_mps_file(const std::filesystem::path &mps_file) {
    MPSData mps;
    std::ifstream file(mps_file);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + mps_file.string());
    }

    std::string line, section;
    while (std::getline(file, line)) {
        trim(line);
        if (line.empty()) continue;

        // Identify new sections
        if (line == "NAME") {
            std::getline(file, line);
            trim(line);
            mps.name = line;
            section.clear();
        } else if (line == "ROWS") {
            section = "ROWS";
        } else if (line == "COLUMNS") {
            section = "COLUMNS";
        } else if (line == "RHS") {
            section = "RHS";
        } else if (line == "BOUNDS") {
            section = "BOUNDS";
        } else if (line == "ENDATA") {
            break;
        } else {
            // Parse sections
            std::istringstream iss(line);
            if (section == "ROWS") {
                std::string rowType, rowName;
                iss >> rowType >> rowName;
                mps.rows.push_back(RowInfo{
                    .name = rowName,
                    .type = convertRowType(rowType)
                });
            } else if (section == "COLUMNS") {
                std::string columnName, rowName;
                double value;
                iss >> columnName >> rowName >> value;
                if (rowName == "'MARKER'") {
                    continue;
                }
                mps.columns[columnName][rowName] = value;
            } else if (section == "RHS") {
                std::string rhsName, rowName;
                double value;
                iss >> rhsName >> rowName >> value;
                mps.rhs[rowName] = value;
            } else if (section == "BOUNDS") {
                // Ignore for now
            }
        }
    }

    return mps;
}

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

template <> struct fmt::formatter<RowInfo::Type>: formatter<string_view> {
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(RowInfo::Type t, FormatContext& ctx) {
    string_view name = "unknown";
    switch (t) {
    case RowInfo::Type::NONE:   name = "None"; break;
    case RowInfo::Type::LESS_THAN: name = "LEQ"; break;
    case RowInfo::Type::GREATER_THAN:   name = "GEQ"; break;
    case RowInfo::Type::EQUAL:   name = "EQ"; break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};

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

    fmt::print("variables: {}\r\n", variables);
    fmt::print("constraints: {}\r\n", constraints);
    fmt::print("constraint_type_from_name: {}\r\n", constraint_type_from_name);
    fmt::print("constraint_start_idx_from_name: {}\r\n", constraint_start_idx_from_name);

    std::vector<Eigen::Triplet<float>> constraint_entries;
    for (int var_idx = 0; var_idx < variables.size(); var_idx++) {
        const std::string &var_name = variables.at(var_idx);
        const auto &entries = mps.columns.at(var_name);
        fmt::print("Entries for {}: {}\r\n", var_name, entries);
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

            for (int i = 0; i < constraint_types.size(); i++) {
                const auto constraint_type = constraint_types.at(i);
                constraint_entries.push_back(
                    {constraint_start_idx + i, var_idx, 
                        constraint_type == CT::LESS_THAN ? value : -value});
            }
        }
    }

    for (int constraint_idx = 0; constraint_idx < constraints.size(); constraint_idx++) {
        const std::string &constraint_name = constraints.at(constraint_idx);
        const auto true_constraint_type = constraint_type_from_name.at(constraint_name);
        if (true_constraint_type == CT::NONE || true_constraint_type == CT::UNKNOWN) {
            continue;
        }
        const float rhs_value = mps.rhs.at(constraint_name);
        const int constraint_start_idx = constraint_start_idx_from_name.at(constraint_name);
        const auto constraint_types = true_constraint_type == CT::EQUAL ? 
            std::vector<CT>{CT::LESS_THAN, CT::GREATER_THAN} 
            : std::vector<CT>{true_constraint_type};
        for (int i = 0; i < constraint_types.size(); i++) {
            const auto constraint_type = constraint_types.at(i);
            out.rhs(constraint_start_idx + i) = constraint_type == CT::LESS_THAN ? rhs_value : -rhs_value;
        }
    }

    out.constraints.setFromTriplets(constraint_entries.begin(), constraint_entries.end());

    return out;
}

