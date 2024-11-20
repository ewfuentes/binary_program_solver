
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

TEST(LoadMpsTest, load_simple_example) {
    MPSData mps = load_mps_file("sample.mps");

    EXPECT_EQ(mps.columns.size(), 10);
    EXPECT_EQ(mps.rows.size(), 3);
}


TEST(LoadMpsTest, load_full_example) {
    MPSData mps = load_mps_file("glass-sc.mps");

    EXPECT_EQ(mps.columns.size(), 214);
    EXPECT_EQ(mps.rows.size(), 6120);
}
