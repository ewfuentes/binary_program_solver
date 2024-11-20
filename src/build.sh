#!/bin/bash

# Output build directory path.
CTR_BUILD_DIR=/build

echo "Building the project..."

# ----------------------------------------------------------------------------------
# ------------------------ PUT YOUR BULDING COMMAND(s) HERE ------------------------
# ----------------------------------------------------------------------------------
# ----- This sctipt is executed inside the development container:
# -----     * the current workdir contains all files from your src/
# -----     * all output files (e.g. generated binaries, test inputs, etc.) must be places into $CTR_BUILD_DIR
# ----------------------------------------------------------------------------------
# Build code.
SRC_DIR=`pwd`
mkdir -p /tmp/build
cmake -GNinja ${SRC_DIR}
cmake --build .
cp -r . ${CTR_BUILD_DIR}
chown -R ${UID}:${UID} ${CTR_BUILD_DIR}
# nvcc -O3 vector_add.cu -o ${CTR_BUILD_DIR}/vector_add
