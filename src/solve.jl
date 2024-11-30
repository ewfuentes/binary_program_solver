using JuMP, Gurobi
problem = "p0201"
file = "src/test_problems/$problem.mps"
model = read_from_file(file)
set_optimizer(model, Gurobi.Optimizer)
optimize!(model)
for v in all_variables(model)
    println(v, ": ", value(v))
end
println(model)