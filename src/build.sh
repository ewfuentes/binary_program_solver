# Output build directory path.
CTR_BUILD_DIR=/build

echo "Building the project..."

# ----------------------------------------------------------------------------------
# ------------------------ PUT YOUR BULDING COMMAND(s) HERE ------------------------
# ----------------------------------------------------------------------------------
nvcc vector_add.cu -o ${CTR_BUILD_DIR}/vector_add
