
#include <gtest/gtest.h>

#include "load_mps.hh"

TEST(LoadMpsTest, hello_world) {
    load_mps("/tmp/test_path.mps");
    EXPECT_TRUE(false);
}
