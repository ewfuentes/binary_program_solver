
#include "load_mps.hh"

#include <fstream>
#include <string>

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
                mps.columns[columnName][rowName] = value;
            } else if (section == "RHS") {
                std::string rhsName, rowName;
                double value;
                iss >> rhsName >> rowName >> value;
                mps.rhs[rowName] = value;
            } else if (section == "BOUNDS") {
                std::string boundType, varName;
                double value;
                iss >> boundType >> varName >> value;
                mps.bounds[varName] = value;
            }
        }
    }

    return mps;
}

