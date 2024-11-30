#include"../gpu_solver.cuh"
int main(){
    cuda::std::array<cuda::std::tuple<int, pair<int>>, 18> const var_2_constr{
cuda::std::make_tuple(0, pair{1, -1}), 
cuda::std::make_tuple(0, pair{3, 1}), 
cuda::std::make_tuple(1, pair{1, 1}), 
cuda::std::make_tuple(1, pair{3, 1}), 
cuda::std::make_tuple(2, pair{0, 1}), 
cuda::std::make_tuple(2, pair{2, -1}), 
cuda::std::make_tuple(3, pair{0, 1}), 
cuda::std::make_tuple(3, pair{2, 1}), 
cuda::std::make_tuple(4, pair{3, 1}), 
cuda::std::make_tuple(4, pair{4, 1}), 
cuda::std::make_tuple(5, pair{3, 1}), 
cuda::std::make_tuple(5, pair{4, -1}), 
cuda::std::make_tuple(6, pair{3, 1}), 
cuda::std::make_tuple(7, pair{0, 1}), 
cuda::std::make_tuple(7, pair{5, -1}), 
cuda::std::make_tuple(8, pair{0, 1}), 
cuda::std::make_tuple(8, pair{5, 1}), 
cuda::std::make_tuple(9, pair{0, 1})
};
    cuda::std::array<cuda::std::tuple<int, pair<int>>, 18> const constr_2_var{
cuda::std::make_tuple(0, pair{2, 1}), 
cuda::std::make_tuple(0, pair{3, 1}), 
cuda::std::make_tuple(0, pair{7, 1}), 
cuda::std::make_tuple(0, pair{8, 1}), 
cuda::std::make_tuple(0, pair{9, 1}), 
cuda::std::make_tuple(1, pair{0, -1}), 
cuda::std::make_tuple(1, pair{1, 1}), 
cuda::std::make_tuple(2, pair{2, -1}), 
cuda::std::make_tuple(2, pair{3, 1}), 
cuda::std::make_tuple(3, pair{0, 1}), 
cuda::std::make_tuple(3, pair{1, 1}), 
cuda::std::make_tuple(3, pair{4, 1}), 
cuda::std::make_tuple(3, pair{5, 1}), 
cuda::std::make_tuple(3, pair{6, 1}), 
cuda::std::make_tuple(4, pair{4, 1}), 
cuda::std::make_tuple(4, pair{5, -1}), 
cuda::std::make_tuple(5, pair{7, -1}), 
cuda::std::make_tuple(5, pair{8, 1})
};
    auto const problem = problem_t<10, 6, 18>{
        .obj = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        .rhs = {5, 0, 0, 5, 0, 0},
        .rhs_n = {5, 2, 2, 5, 2, 2},
        .is_eq = bitset<6>::from(cuda::std::array<bool, 6>{false, false, false, false, false, false}),
        .var_2_constr = range_array<pair<int>, 10, 18>(var_2_constr),
        .constr_2_var = range_array<pair<int>, 6, 18>(constr_2_var),
    };
    solve_gpu_impl(problem);
}

