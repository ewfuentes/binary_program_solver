
using JuMP
import LinearAlgebra

function create_n_queens(N)
    model = Model()

    @variable(model, x[1:N, 1:N], Bin)

    for i in 1:N
        # There can only be a single queen in each row
        @constraint(model, sum(x[i, :]) == 1)
        # There can only be a single queen in each column 
        @constraint(model, sum(x[:, 1]) == 1)
    end

    for i in -(N - 1):(N-1)
        @constraint(model, sum(LinearAlgebra.diag(x, i)) <= 1)
        @constraint(model, sum(LinearAlgebra.diag(reverse(x; dims = 1), i)) <= 1)
    end

    # break symmetry by preferring to place queens in the top left corner
    @objective(model, Min, sum((i * N + j) * x[i, j] for i in 1:N for j in 1:N))

    @show model
    return model
end

function generate_cpp(model)
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


    return """#include"../gpu_solver.cuh"
    int main(){
        $cpp_var_2_constr
        $cpp_constr_2_var
        auto const problem = problem_t<$(length(variables)), $(length(constraints)), $(length(triplets))>{
            .obj = {$cpp_obj},
            .rhs = {$cpp_rhs},
            .rhs_n = {$cpp_rhs_n},
            .is_eq = bitset<$(length(constraints))>::from(cuda::std::array<bool, $(length(constraints))>{$(join(string.(is_eq), ", "))}),
            .var_2_constr = range_array<pair<int>, $(length(variables)), $(length(triplets))>(var_2_constr),
            .constr_2_var = range_array<pair<int>, $(length(constraints)), $(length(triplets))>(constr_2_var),
        };
        solve_gpu_impl(problem);
    }
    """
end

for n in 4:4:16
    model = create_n_queens(n)
    file_contents = generate_cpp(model)
    open("src/converted/queens_$n.cu", "w") do io
        println(io, file_contents)
    end
    
    write_to_file(model, "src/test_problems/queens_$n.mps")
end

