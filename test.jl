using JuMP, GLPK
model = read_from_file("src/test_problems/sample.mps")
set_optimizer(model, GLPK.Optimizer)
optimize!(model)

objective_value(model)
value.(all_variables(model))