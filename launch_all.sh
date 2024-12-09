for i in `find src/test_problems -name "*.mps"`; do
    for j in gurobi gurobi_nerfed glpk glpk_nerfed; do
        echo "Running $i with $j"
        sbatch launch.sh julia bench.jl "$i" "$j"
    done
done