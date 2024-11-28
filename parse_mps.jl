using JuMP, Gurobi

file = "src/test_problems/stein15inf.mps"
model = read_from_file(file)
# set_optimizer(model, Gurobi.Optimizer)
# optimize!(model)
# objective_sense(model)

# model = Model(Gurobi.Optimizer)
# @variable(model, x[1:4], Bin)
# @constraint(model, x[1] + x[2] == 1)
# @constraint(model, x[3] + x[4] == 1)
# @objective(model, Min, x[1] + x[2] + x[3] + x[4])

println(model)
let
    constraints = Set(all_constraints(model, include_variable_in_set_constraints=false))
    variables = Set(all_variables(model))
    varx = copy(variables)
    c = first(constraints)
    coeffs = Dict(
        c => Dict(i => normalized_coefficient(c, i) for i in varx if normalized_coefficient(c, i) != 0) for c in constraints
    )
    ordered_vars = VariableRef[]
    while !isempty(varx)
        c, coeff = argmin(coeffs) do (c, coeff)
            length(coeff)
        end
        @show c, length(coeff)
        if length(coeff) == 0
            delete!(coeffs, c)
            continue
        end

        x = first(keys(coeff))
        push!(ordered_vars, x)

        delete!(varx, x)
        for (c, coeff) in coeffs
            delete!(coeff, x)
        end
    end
    ordered_vars
    var_to_idx=  Dict(v => i - 1 for (i, v) in enumerate(ordered_vars))
    constr_to_idx = Dict(c => i - 1 for (i, c) in enumerate(constraints))
    triplets = Tuple{Int, Int, Int}[]
    @show constraints
    is_eq = Vector{Bool}(undef, length(constraints))
    rhs = Vector{Int32}(undef, length(constraints))
    for c in constraints
        o = constraint_object(c)
        negate = 1
        if isa(o.set, MOI.EqualTo{Float64})
            is_eq[constr_to_idx[c] + 1] = true
        elseif isa(o.set, MOI.LessThan{Float64})
            is_eq[constr_to_idx[c] + 1] = false
        elseif isa(o.set, MOI.GreaterThan{Float64})
            is_eq[constr_to_idx[c] + 1] = false
            negate = -1
        else
            error("Unsupported constraint type $o $(o.set)")
        end
        rhs[constr_to_idx[c] + 1] = negate * normalized_rhs(c)
            
        for v in variables
            if normalized_coefficient(c, v) != 0
                push!(triplets, (var_to_idx[v], constr_to_idx[c], negate * normalized_coefficient(c, v)))
            end
        end
    end
    triplets = sort(triplets, by = x -> (x[1], x[2]))
    cpp_var_2_constr = "cuda::std::array<cuda::std::tuple<int, pair<int>>, $(length(triplets))> const var_2_constr{\n" * join(["cuda::std::make_tuple($v, pair{$c, $coeff})" for (v, c, coeff) in triplets], ", \n") * "\n};"
    triplets = sort(triplets, by = x -> (x[2], x[1]))
    cpp_var_2_constr = "cuda::std::array<cuda::std::tuple<int, pair<int>>, $(length(triplets))> const constr_2_var{\n" * join(["cuda::std::make_tuple($c, pair{$v, $coeff})" for (v, c, coeff) in triplets], ", \n") * "\n};"
    obj_terms = objective_function(model).terms
    if objective_sense(model) != MOI.MIN_SENSE
        @assert objective_sense(model) == MOI.MAX_SENSE
        obj_terms = Dict(v => -c for (v, c) in obj_terms)
    end

    cpp_obj = join([string(get(obj_terms, v, 0)) for v in ordered_vars], ", ")
    cpp_rhs = join(string.(rhs), ", ")
    rhs_n = zeros(Int, length(constraints))
    for (v, c, coeff) in triplets
        rhs_n[c + 1] += 1
    end
    cpp_rhs_n = join(string.(rhs_n), ", ")
    

    println("""$cpp_var_2_constr
    $cpp_var_2_constr
      auto const problem = problem_t<4, 2, 4>{
      .obj = {$cpp_obj},
      .rhs = {$cpp_rhs},
      .rhs_n = {$cpp_rhs_n},
      .is_eq = bitset<$(length(constraints))>::from(cuda::std::array<bool, $(length(constraints))>{$(join(string.(is_eq), ", "))}),
      .var_2_constr = range_array<pair<int>, $(length(variables)), $(length(triplets))>::from_tiplet(var_2_constr),
      .constr_2_var = range_array<pair<int>, $(length(constraints)), $(length(triplets))>::from_tiplet(constr_2_var),
    """)
end

print(model)