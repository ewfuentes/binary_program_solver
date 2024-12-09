using JuMP
using Gurobi
using GLPK
using JSON
function make_opt(opt)
    @show opt
    if opt == "glpk"
        return optimizer_with_attributes(GLPK.Optimizer, "msg_lev" => GLPK.GLP_MSG_ALL)
    elseif opt == "glpk_nerfed"
        opt == "glpk_nerfed"
        return optimizer_with_attributes(GLPK.Optimizer, "presolve" => 0,  "msg_lev" => GLPK.GLP_MSG_ALL, "br_tech" => GLPK.GLP_BR_FFV, "pp_tech" => GLPK.GLP_PP_NONE, "sr_heur" => GLPK.GLP_OFF, "mir_cuts" => GLPK.GLP_OFF, "gmi_cuts" => GLPK.GLP_OFF, "cov_cuts" => GLPK.GLP_OFF, "clq_cuts" => GLPK.GLP_OFF)
    elseif opt == "gurobi"
        return Gurobi.Optimizer
    elseif opt == "gurobi_nerfed"
        return optimizer_with_attributes(Gurobi.Optimizer, "Presolve" => 0, "Cuts" => 0, "Heuristics" => 0, "Symmetry" => 0)
    # elseif opt == "cplex"
    #     return CPLEX.Optimizer
    # elseif opt == "cplex_nerfed"
    #     return optimizer_with_attributes(CPLEX.Optimizer, "PreInd" => 0, "AggInd" => 0, "HeurInd" => 0, "Symmetry" => 0, "CutsFactor" => 0)
    end
    raise("Unknown optimizer")
end
function solve(file, opt)
    @show file, opt
    start_time = time()
    model = read_from_file(file)
    set_optimizer(model, make_opt(opt))
    set_time_limit_sec(model, 60 * 20)
    optimize!(model)
    x = Dict(
        "solver"=> opt,
        "file" => file,
        "time" => time() - start_time,
        "lower_bound" => objective_bound(model),
        "upper_bound" => objective_bound(model),
        "status" => termination_status(model),
        "n_variables" => length(all_variables(model)),
        "n_constraints" => length(all_constraints(model, include_variable_in_set_constraints=false)),
        "n_nonzeros_obj" => sum(1 for (_, v) in objective_function(model).terms if v != 0),
        "n_nonzeros_constr" => sum(1 for c in all_constraints(model, include_variable_in_set_constraints=false) for v in constraint_object(c).func if v != 0)
    )
    if termination_status(model) == MOI.OPTIMAL
        x["objective"] = objective_value(model)
    end
    return x
end

if abspath(PROGRAM_FILE) == @__FILE__
    
    if length(ARGS) != 2
        println("Usage: $(PROGRAM_FILE) <file> <opt>")
        exit(1)
    end
    _, fn = splitdir(ARGS[1])
    fn, _ = splitext(fn)
    @show fn

    x = solve(ARGS[1], ARGS[2])

    open(joinpath("src/solved/$(fn)_$(ARGS[2]).json"), "w") do f
        JSON.print(f, x)
    end
    
end