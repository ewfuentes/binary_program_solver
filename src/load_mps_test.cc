
#include <gtest/gtest.h>

#include "load_mps.hh"

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

TEST(LoadMpsTest, hello_world) {
    MPSData mps = load_mps_file("sample.mps");
    std::cout << "Parsed MPS File: " << mps.name << std::endl;

    std::cout << "\nRows (Constraints):" << std::endl;
    for (const auto &row : mps.rows) {
        std::cout << "  " << row.name << " " << to_string(row.type) << std::endl;
    }

    std::cout << "\nColumns (Variables):" << std::endl;
    for (const auto &col : mps.columns) {
        std::cout << "  " << col.first << ":";
        for (const auto &rowVal : col.second) {
            std::cout << " (" << rowVal.first << ", " << rowVal.second << ")";
        }
        std::cout << std::endl;
    }

    std::cout << "\nRHS:" << std::endl;
    for (const auto &rhs : mps.rhs) {
        std::cout << "  " << rhs.first << ": " << rhs.second << std::endl;
    }

    std::cout << "\nBounds:" << std::endl;
    for (const auto &bound : mps.bounds) {
        std::cout << "  " << bound.first << ": " << bound.second << std::endl;
    }
}
