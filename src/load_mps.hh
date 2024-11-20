
#pragma once

#include <filesystem>
#include <unordered_map>
#include <vector>

struct RowInfo {
    enum class Type {
        NONE,  // N: No constraint, used for the objective function
        LESS_THAN,  // L: Less than or equal to
        GREATER_THAN,  // G: Greater than or equal to
        EQUAL,  // E: Equality
        UNKNOWN // For invalid or unsupported row types
    };
    std::string name;
    Type type;
};

struct MPSData {
    std::string name;
    std::vector<RowInfo> rows;                   // Constraints (e.g., L, G, N)
    std::unordered_map<std::string, std::unordered_map<std::string, double>> columns; // Variables and coefficients
    std::unordered_map<std::string, double> rhs;               // RHS values
};


MPSData load_mps_file(const std::filesystem::path &mps_file);
