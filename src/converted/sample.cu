#include"../gpu_solver.h"
int main(int argc, char **argv) {
    cuda::std::array<cuda::std::tuple<int, pair<int>>, 18> const var_2_constr{
cuda::std::make_tuple(0, pair{0, -1}), 
cuda::std::make_tuple(0, pair{4, 1}), 
cuda::std::make_tuple(1, pair{0, 1}), 
cuda::std::make_tuple(1, pair{4, 1}), 
cuda::std::make_tuple(2, pair{1, 1}), 
cuda::std::make_tuple(2, pair{5, 1}), 
cuda::std::make_tuple(3, pair{1, -1}), 
cuda::std::make_tuple(3, pair{5, 1}), 
cuda::std::make_tuple(4, pair{2, -1}), 
cuda::std::make_tuple(4, pair{4, 1}), 
cuda::std::make_tuple(5, pair{2, 1}), 
cuda::std::make_tuple(5, pair{4, 1}), 
cuda::std::make_tuple(6, pair{4, 1}), 
cuda::std::make_tuple(7, pair{3, -1}), 
cuda::std::make_tuple(7, pair{5, 1}), 
cuda::std::make_tuple(8, pair{3, 1}), 
cuda::std::make_tuple(8, pair{5, 1}), 
cuda::std::make_tuple(9, pair{5, 1})
};
    cuda::std::array<cuda::std::tuple<int, pair<int>>, 18> const constr_2_var{
cuda::std::make_tuple(0, pair{0, -1}), 
cuda::std::make_tuple(0, pair{1, 1}), 
cuda::std::make_tuple(1, pair{2, 1}), 
cuda::std::make_tuple(1, pair{3, -1}), 
cuda::std::make_tuple(2, pair{4, -1}), 
cuda::std::make_tuple(2, pair{5, 1}), 
cuda::std::make_tuple(3, pair{7, -1}), 
cuda::std::make_tuple(3, pair{8, 1}), 
cuda::std::make_tuple(4, pair{0, 1}), 
cuda::std::make_tuple(4, pair{1, 1}), 
cuda::std::make_tuple(4, pair{4, 1}), 
cuda::std::make_tuple(4, pair{5, 1}), 
cuda::std::make_tuple(4, pair{6, 1}), 
cuda::std::make_tuple(5, pair{2, 1}), 
cuda::std::make_tuple(5, pair{3, 1}), 
cuda::std::make_tuple(5, pair{7, 1}), 
cuda::std::make_tuple(5, pair{8, 1}), 
cuda::std::make_tuple(5, pair{9, 1})
};
    std::string self = "sample";
    using namespace std::literals;
    std::chrono::seconds duration = 1min;
    if (argc >= 2){
        duration = std::chrono::minutes(std::stoi(argv[1]));
    }
    auto const config = run_config{
        .self = self,
        .duration = duration
    };
    auto const problem = problem_t<10, 6, 18>{
        .obj = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        .rhs = {0, 0, 0, 0, 5, 5},
        .rhs_n = {2, 2, 2, 2, 5, 5},
        .is_eq = bitset<6>::from(cuda::std::array<bool, 6>{false, false, false, false, false, false}),
        .var_2_constr = range_array<pair<int>, 10, 18>(var_2_constr),
        .constr_2_var = range_array<pair<int>, 6, 18>(constr_2_var),
    };
    solve_gpu_impl(problem, config);
}

