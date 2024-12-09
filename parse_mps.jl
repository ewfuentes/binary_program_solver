using JuMP, Gurobi

for problem = [
    "glass-sc",
    "p0201",
    "p2m2p1m1p0n100",
    "pb-market-split8-70-4",
    "sample",
    "stein15inf",
    "stein45inf",
    # "stein9inf"
    ]
    file = "src/test_problems/$problem.mps"
    model = read_from_file(file)
    # set_optimizer(model, Gurobi.Optimizer)
    # optimize!(model)
    let of = objective_function(model)
        for (iidx, i) in enumerate(all_variables(model)),
            (jidx, j) in enumerate(all_variables(model))

            if iidx < jidx && get(of.terms, i, 0) == get(of.terms, j, 0)
                sym = true
                for c in all_constraints(model, include_variable_in_set_constraints=false)
                    if normalized_coefficient(c, i) != normalized_coefficient(c, j)
                        sym = false
                        # @show i, j, c
                        break
                    end
                end
                if sym
                    @show i, j
                    @constraint(model, i <= j)
                end
            end
        end
    end

    for c in all_constraints(model, include_variable_in_set_constraints=false)
        o = constraint_object(c)
        if isa(o.set, MOI.Interval{Float64})
            
            l = o.set.lower
            u = o.set.upper
            if l != -Inf
                add_constraint(model, ScalarConstraint(o.func, MOI.LessThan(l)))
            end
            if u != Inf
                add_constraint(model, ScalarConstraint(o.func, MOI.GreaterThan(u)))
            end

            delete(model, c)
        end
    end

    for c in all_constraints(model, include_variable_in_set_constraints=false)
        o = constraint_object(c)
        if isa(o.set, MOI.GreaterThan{Float64})
            
            add_constraint(model, ScalarConstraint(-o.func, MOI.LessThan(-o.set.lower)))
            delete(model, c)
        end
    end

    for i in all_constraints(model, include_variable_in_set_constraints=false)
        @show i
    end
    cpp = let
        constraints = Set(all_constraints(model, include_variable_in_set_constraints=false))
        variables = Set(all_variables(model))
        varx = copy(variables)
        c = first(constraints)
        coeffs = Dict(
            c => Dict(i => normalized_coefficient(c, i) for i in varx if normalized_coefficient(c, i) != 0) for c in constraints
        )
        weights = objective_function(model).terms
        ordered_vars = VariableRef[]
        while !isempty(varx)
            c, coeff = argmin(coeffs) do (c, coeff)
                length(coeff)
            end

            if length(coeff) == 0
                delete!(coeffs, c)
                continue
            end

            ox = argmax(varx) do x
                abs(get(weights, x, 0))
            end
            
            x = if length(coeff) <= 5 || get(weights, ox, 0) == 0
                # @show c, length(coeff)
                x = argmax(keys(coeff)) do x
                    abs(get(weights, x, 0))
                end
            else 
                ox
            end

            push!(ordered_vars, x)

            delete!(varx, x)
            for (c, coeff) in coeffs
                delete!(coeff, x)
            end
        end
        
        println("ordering: ", ordered_vars)

        var_to_idx = Dict(v => i - 1 for (i, v) in enumerate(ordered_vars))
        constr_to_idx = Dict(c => i - 1 for (i, c) in enumerate(constraints))
        triplets = Tuple{Int,Int,Int}[]
        # @show constraints
        is_eq = Vector{Bool}(undef, length(constraints))
        rhs = Vector{Int32}(undef, length(constraints))
        for c in constraints
            o = constraint_object(c)
            negate = 1
            if isa(o.set, MOI.EqualTo{Float64})
                is_eq[constr_to_idx[c]+1] = true
            elseif isa(o.set, MOI.LessThan{Float64})
                is_eq[constr_to_idx[c]+1] = false

            else
                error("unsupported constraint type")

            end


            rhs[constr_to_idx[c]+1] = negate * normalized_rhs(c)

            for v in variables
                if normalized_coefficient(c, v) != 0
                    push!(triplets, (var_to_idx[v], constr_to_idx[c], negate * normalized_coefficient(c, v)))
                end
            end
        end
        triplets = sort(triplets, by=x -> (x[1], x[2]))
        cpp_var_2_constr = "cuda::std::array<cuda::std::tuple<int, pair<int>>, $(length(triplets))> const var_2_constr{\n" * join(["cuda::std::make_tuple($v, pair{$c, $coeff})" for (v, c, coeff) in triplets], ", \n") * "\n};"
        triplets = sort(triplets, by=x -> (x[2], x[1]))
        cpp_constr_2_var = "cuda::std::array<cuda::std::tuple<int, pair<int>>, $(length(triplets))> const constr_2_var{\n" * join(["cuda::std::make_tuple($c, pair{$v, $coeff})" for (v, c, coeff) in triplets], ", \n") * "\n};"
        obj_terms = objective_function(model).terms
        if objective_sense(model) != MOI.MIN_SENSE
            @assert objective_sense(model) == MOI.MAX_SENSE
            obj_terms = Dict(v => -c for (v, c) in obj_terms)
        end

        cpp_obj = join([string(Int(get(obj_terms, v, 0))) for v in ordered_vars], ", ")
        cpp_rhs = join(string.(rhs), ", ")
        rhs_n = zeros(Int, length(constraints))
        for (v, c, coeff) in triplets
            rhs_n[c+1] += 1
        end
        cpp_rhs_n = join(string.(rhs_n), ", ")


        """#include"../gpu_solver.h"
    int main(int argc, char **argv) {
        $cpp_var_2_constr
        $cpp_constr_2_var
        std::string self = "$problem";
        using namespace std::literals;
        std::chrono::seconds duration = 1min;
        if (argc >= 2){
            duration = std::chrono::minutes(std::stoi(argv[1]));
        }
        auto const config = run_config{
            .self = self,
            .duration = duration
        };
        auto const problem = problem_t<$(length(variables)), $(length(constraints)), $(length(triplets))>{
            .obj = {$cpp_obj},
            .rhs = {$cpp_rhs},
            .rhs_n = {$cpp_rhs_n},
            .is_eq = bitset<$(length(constraints))>::from(cuda::std::array<bool, $(length(constraints))>{$(join(string.(is_eq), ", "))}),
            .var_2_constr = range_array<pair<int>, $(length(variables)), $(length(triplets))>(var_2_constr),
            .constr_2_var = range_array<pair<int>, $(length(constraints)), $(length(triplets))>(constr_2_var),
        };
        solve_gpu_impl(problem, config);
    }
    """
    end

    open("src/converted/$problem.cu", "w") do io
        println(io, cpp)
    end

end
