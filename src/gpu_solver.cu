
#include "gpu_solver.hh"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <type_traits>
#ifdef CLANG
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>

#endif

#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>
#include <cuda/std/array>
#include <cuda/std/tuple>
#include <fmt/format.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <limits>
#include <tuple>
#include <vector>

#define STR_DETAIL(x) #x
#define STR(x) STR_DETAIL(x)

#define cuda_error(code)                                                       \
  {                                                                            \
    if ((code) != cudaSuccess) [[unlikely]] {                                  \
      fprintf(stderr,                                                          \
              "cuda error %s in file " __FILE__ ":" STR(__LINE__) "\n",        \
              cudaGetErrorString(code));                                       \
      if (abort)                                                               \
        throw std::runtime_error("assertion failed");                          \
      ;                                                                        \
    }                                                                          \
  }

template <typename T> using pair = cuda::std::array<T, 2>;

template <typename T> T constexpr cdiv(T a, T b) { return (a + b - 1) / b; }

template <unsigned int n> struct bitset {
  uint32_t data[cdiv(n, 32u)];

  template <typename T> bool constexpr get(T i) const {
    return (data[i / 32u] >> (i % 32)) & 1;
  }
  template <typename T> void constexpr set(T i, bool b) {
    if (b)
      data[i / 32u] |= 1u << (i % 32u);
    else
      data[i / 32u] &= ~(1u << (i % 32u));
  }

  static bitset<n> __host__ full(bool b) {
    bitset<n> res;
    for (int i = 0; i < cdiv(n, 32u); i++) {
      res.data[i] = b ? ~0u : 0u;
    }
    return res;
  }
};
template <typename T, int x, int y> struct range_array {
  cuda::std::array<int, x + 1> idx;
  cuda::std::array<T, y> val;

  auto static from_triplet(
      cuda::std::array<cuda::std::tuple<int, T>, y> const &triplet) {
    cuda::std::array<int, x + 1> delta;
    range_array res;

    for (auto &i : delta)
      i = 0;

    for (auto &i : res.idx)
      i = 0;

    // Count the number of items that have a particular row index
    for (auto &[i, _] : triplet)
      ++res.idx[i + 1];

    // compute prefix sums to get row start indices
    for (int i = 0; i < x; i++)
      res.idx[i + 1] += res.idx[i];

    // Assign the items so that each value is in the correct row, though unsorted
    for (auto &[i, j] : triplet) {
      res.val[res.idx[i] + delta[i]] = j;
      ++delta[i];
    }

    // Check that items were assigned correctly
    for (int i = 0; i < x; ++i) {
      if (res.idx[i] + delta[i] != res.idx[i + 1]) {
        fmt::println("delta: {}", fmt::join(delta, ", "));
        fmt::println("idx: {}", fmt::join(res.idx, ", "));
        fmt::println("triplet: {}", fmt::join(triplet, ", "));

        throw std::runtime_error("range array construction failed");
      }
    }

    // sort the items in the rows. Note this only works if items
    // of type T can be sorted. In our case, we're using std::arrays
    // which are compared lexographically
    for (int i = 0; i < x; ++i) {
      std::stable_sort(res.val.begin() + res.idx[i],
                       res.val.begin() + res.idx[i + 1]);
    }
    return res;
  }
};

template <int n, int m, int n_terms> struct problem_t {
  auto static constexpr n_var = n;
  auto static constexpr n_constr = m;
  cuda::std::array<int, n> obj;
  cuda::std::array<int, m> rhs;
  // This tracks the number of variable in each constraint that have already
  // been assigned.
  cuda::std::array<int, m> rhs_n;
  bitset<m> is_eq;
  // first item is constraint index, second item is coefficient
  range_array<pair<int>, n, n_terms> var_2_constr;
  // first item is variable index, second item is coefficient
  range_array<pair<int>, m, n_terms> constr_2_var;
};

template <int n_var, int n_constr> struct alignas(8) solution_t {
  int index;
  int upper_bound;
  int obj;
  bitset<n_var * 2> var;

  cuda::std::array<int, n_constr> rhs;
  cuda::std::array<int, n_constr> rhs_n;
};
// template <typename T> union data_u {
//   static_assert(sizeof(T) % 4 == 0);
//   T x;
//   uint32_t data[sizeof(T) / 32];
// };

// Must include after definition of problem_t and solution_t
#include "gpu_solver_formatter.hh"

template <int NUM_VARS, int NUM_CONSTRAINTS, int NUM_NONZERO>
struct GPUSolverMemory {
  problem_t<NUM_VARS, NUM_CONSTRAINTS, NUM_NONZERO> *problem;

  solution_t<NUM_VARS, NUM_CONSTRAINTS> *queue;
  size_t queue_size;
  size_t queue_max_size;

  solution_t<NUM_VARS, NUM_CONSTRAINTS> *best_solution;

  solution_t<NUM_VARS, NUM_CONSTRAINTS> *delta_queue;
  uint32_t *delta_mask;
  uint32_t *delta_cumsum;
  size_t delta_queue_size;

  void *scan_workspace;
  size_t scan_workspace_size;
};

template <typename T> __device__ void mycpy(T const *src, T *dst) {

  using I = uint32_t;
  static_assert(sizeof(T) % sizeof(I) == 0);
  I *Isrc = reinterpret_cast<I *>((void *)src);
  I *Idst = reinterpret_cast<I *>((void *)dst);
  for (int i = threadIdx.x; i < sizeof(T) / sizeof(I); i += blockDim.x) {
    Idst[i] = Isrc[i];
  }
}

template <bool DEBUG, int n_threads, int n_var, int n_constr, int n_terms>
__global__ void
traverse(problem_t<n_var, n_constr, n_terms> const *const problem,
         solution_t<n_var, n_constr> const *const queue, // SDF
         uint32_t *delta_mask,                           // SDF
         solution_t<n_var, n_constr> *delta_queue) {
  using sol_t = solution_t<n_var, n_constr>;
  sol_t const *__restrict__ const qel = queue + blockIdx.x;
  __shared__ bool kill_switch;
  __shared__ sol_t cur;
  __shared__ sol_t next;

  // Each block has access to n_threads to compute some successors starting
  // from a single queued item
  // Copy the queued item to shared memory
  mycpy(qel, &cur);
  __syncthreads();
  // If the current item has been fully assigned, don't queue any more items
  // if (cur.index >= problem->n_var) {
  if (cur.index >= n_var) { // this is known at compile time, so save a read
    if (threadIdx.x <= 1) {
      delta_mask[2 * blockIdx.x + threadIdx.x] = 0;
    }
    return;
  }

  // Iterate over possible assignments to the next variable
  for (auto val : {false, true}) {
    __syncthreads();
    auto const dqidx = 2 * blockIdx.x + val;
    // Seems like all threads write this value. Is that problematic?
    // My guess is that it isn't because of the __syncthreads.
    kill_switch = false;

    // Check if the current assignment has been previously ruled out
    if (!cur.var.get(cur.index * 2 + val)) {
      // Since all threads read the same value, there is no divergence across warps here.
      if (threadIdx.x == 0) {
        delta_mask[dqidx] = 0;
      }
      continue;
    }
    __syncthreads();
    // Make a copy of the current partial solution
    mycpy(&cur, &next);
    __syncthreads();
    // Is there any benefit to doing the different writes in different threads?
    // I don't think there is...
    if (threadIdx.x == 0)
      next.var.set(cur.index * 2, val == false);
    if (threadIdx.x == 1)
      next.var.set(cur.index * 2 + 1, val == true);
    if (threadIdx.x == 2)
      next.index++;
    __syncthreads();
    auto const var_idx = cur.index;
    // Iterate over the constraints that contain the newly assigned variable
    // The contstraints are processed in parallel
    auto const var_begin = problem->var_2_constr.idx[var_idx];
    auto const var_end = problem->var_2_constr.idx[var_idx + 1];
    for (uint32_t i = threadIdx.x + var_begin; i < var_end && !kill_switch;
         i += blockDim.x) {
      // Note to self: this is an uncoalesced read. Is the problem also small enough
      // to read into shared memory?
      auto const [constr_idx, coeff] = problem->var_2_constr.val[i];
      // Remove the assigned variable from the constraint by updating the RHS
      next.rhs[constr_idx] -= coeff * val;
      // Update the number of assigned variables for this constraint. 
      // Note to self: if we change the meaning of rhs_n to be the number of variables
      // left to assign and decrement instead, we could save a read of the expected
      // number of variables per constraint from the problem
      ++next.rhs_n[constr_idx];
      auto const rhs_n = next.rhs_n[constr_idx];
      auto const rhs_exp = problem->rhs_n[constr_idx];
      auto const is_eq = problem->is_eq.get(constr_idx);
      if (rhs_n == rhs_exp) {
        if ((is_eq && next.rhs[constr_idx] != 0) ||
            (!is_eq && next.rhs[constr_idx] > 0)) {
          // The current partial assignment violates a constraint
          // stop checking constraints and bailout
          kill_switch = true;
          break;
        }
      }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      if (kill_switch) {
        next.obj = std::numeric_limits<int>::max();
      } else {
        // Update the realized cost
        next.obj += problem->obj[cur.index] * val;
      }
    }
    if constexpr (DEBUG) {
      if (threadIdx.x == 0) {
        printf("blockidx %d, val %d, kill_switch %d dqidx %d\n", blockIdx.x, val,
               kill_switch, dqidx);
      }
    }

    if (threadIdx.x == 0) {
      // Note that if the assignment has been completely assigned, we set delta_mask to 0
      delta_mask[dqidx] = !kill_switch && (next.index < n_var);
    }
    mycpy(&next, delta_queue + dqidx);
    // I wonder if this syncthreads is required given that the loop starts with one
    // Perhaps it's a good habit to do so after copying using mycpy
    __syncthreads();
  }
}

template <uint32_t n_threads, int n_var, int n_constr>
__global__ void push_back(uint32_t *delta_cumsum,
                          solution_t<n_var, n_constr> const *const delta_queue,
                          solution_t<n_var, n_constr> *queue) {
  using sol_t = solution_t<n_var, n_constr>;
  auto const cur_idx = delta_cumsum[blockIdx.x];
  auto const prev_idx = blockIdx.x == 0 ? 0 : delta_cumsum[blockIdx.x - 1];

  if (cur_idx == prev_idx) {
    return;
  }
  mycpy(&delta_queue[blockIdx.x], &queue[cur_idx - 1]);
}

template <typename T> __device__ T broadcast(T const &x) {
  __shared__ T s;
  if (threadIdx.x == 0) {
    s = x;
  }
  __syncthreads();
  return s;
}
template <bool DEBUG, uint32_t n_threads, int n_var, int n_constr>
__global__ void
update_bounds(solution_t<n_var, n_constr> const *const delta_queue,
              uint32_t *delta_mask, uint32_t delta_q_size,
              solution_t<n_var, n_constr> *best) {
  uint32_t best_idx = 0;
  int best_val = std::numeric_limits<int>::max();
  for (int i = threadIdx.x; i < delta_q_size; i += blockDim.x) {
    if constexpr (DEBUG) {
      printf("delta_queue[%d].index %d obj %d\n", i, delta_queue[i].index,
             delta_queue[i].obj);
    }
    if (delta_queue[i].index == n_var && delta_queue[i].obj < best_val) {
      best_idx = i;
      best_val = delta_queue[i].obj;
    }
  }
  if constexpr (DEBUG) {
    if (best_idx != 0 || best_val != std::numeric_limits<int>::max()) {
      printf("threadIdx.x %d, best_idx %d best_val %d\n", threadIdx.x, best_idx,
             best_val);
    }
  }
  cuda::std::pair<int, uint32_t> candidate {best_val, best_idx};
  using BlockReduce =
      cub::BlockReduce<cuda::std::pair<int, uint32_t> , n_threads>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  auto [r_best_val, r_best_idx] =
      broadcast(BlockReduce(temp_storage)
                    .Reduce(candidate,
                            [](auto const &a, auto const &b) {
                              return a.first < b.first ? a : b;
                            }));
  if constexpr (DEBUG) {
    if (threadIdx.x == 0) {
      printf("r_best_val %d, r_best_idx %d\n", r_best_val, r_best_idx);
    }
  }

  if (r_best_val < best->obj) {
    mycpy(&delta_queue[r_best_idx], best);
  }
}

template <int NUM_VARS, int NUM_CONSTRAINTS> 
std::unordered_map<std::string, int>
get_ordered_vars_and_constraints(const MPSData &mps_data) {
  std::unordered_map<std::string, int> out;

  int idx = 0;
  for (int i = 0; i < static_cast<int>(mps_data.rows.size()); i++) {
    const auto &row = mps_data.rows.at(i);
    if (row.type == RowInfo::Type::NONE) {
      continue; 
    }
    out[row.name] = idx;
    idx++;
  }

  int i = 0;
  for (const auto &[name, _] : mps_data.columns) {
    out[name] = i;
    i++;
  }

  return out;
}

template <int NUM_VARS, int NUM_CONSTRAINTS, int NUM_NONZERO>
problem_t<NUM_VARS, NUM_CONSTRAINTS, NUM_NONZERO> problem_from_mps(const MPSData &mps_data) {
  using TripletArray = cuda::std::array<cuda::std::tuple<int, pair<int>>, NUM_NONZERO>;

  using VarMap = range_array<pair<int>, NUM_VARS, NUM_NONZERO>;
  using ConstraintMap = range_array<pair<int>, NUM_CONSTRAINTS, NUM_NONZERO>;

  TripletArray constraint_from_var;
  TripletArray var_from_constraint;
  cuda::std::array<int, NUM_VARS> objective; 
  cuda::std::array<int, NUM_CONSTRAINTS> rhs;
  cuda::std::array<int, NUM_CONSTRAINTS> n_rhs;
  bitset<NUM_CONSTRAINTS> is_eq;

  n_rhs.fill(0);

  const auto idx_from_name = get_ordered_vars_and_constraints<NUM_VARS, NUM_CONSTRAINTS>(mps_data);
  std::unordered_map<std::string, RowInfo::Type> constraint_type_from_name;
  std::transform(mps_data.rows.begin(), mps_data.rows.end(),
      std::inserter(constraint_type_from_name, constraint_type_from_name.end()),
      [](const auto &row){ return std::pair{row.name, row.type}; });


  int triplet_idx = 0;
  for (const auto &[var_name, coeff_from_constraint]: mps_data.columns) {
    const int var_idx = idx_from_name.at(var_name);
    for (const auto &[constraint_name, coeff] : coeff_from_constraint) {
      const auto constraint_type = constraint_type_from_name.at(constraint_name);
      if (constraint_type == RowInfo::Type::NONE) {
        // This is the objective
        objective[var_idx] = coeff;
        continue;
      }
      const int constraint_idx = idx_from_name.at(constraint_name);
      const int coeff_to_store = constraint_type == RowInfo::Type::GREATER_THAN ? -coeff : coeff;
      constraint_from_var[triplet_idx] = {var_idx, {constraint_idx, coeff_to_store}};
      var_from_constraint[triplet_idx] = {constraint_idx, {var_idx, coeff_to_store}};
      n_rhs[constraint_idx]++;
      triplet_idx++;
    }
  }

  for (const auto &[constraint_name, type] : constraint_type_from_name) {
    if (type == RowInfo::Type::NONE) {
      continue;
    }
    const auto iter = mps_data.rhs.find(constraint_name);
    const int coeff = iter != mps_data.rhs.end() ? iter->second : 0.0;
    const int constraint_idx = idx_from_name.at(constraint_name);
    const int coeff_to_store = type == RowInfo::Type::GREATER_THAN ? -coeff : coeff;
    rhs[constraint_idx] = coeff_to_store;
    is_eq.set(constraint_idx, type == RowInfo::Type::EQUAL);
  }


  return problem_t<NUM_VARS, NUM_CONSTRAINTS, NUM_NONZERO>{
    .obj = objective,
    .rhs = rhs,
    .rhs_n = n_rhs,
    .is_eq = is_eq,
    .var_2_constr = VarMap::from_triplet(constraint_from_var),
    .constr_2_var = ConstraintMap::from_triplet(var_from_constraint),
  };
}

template <int NUM_VARS, int NUM_CONSTRAINTS, int NUM_NONZERO>
GPUSolverMemory<NUM_VARS, NUM_CONSTRAINTS, NUM_NONZERO> allocate_gpu_memory(
  const problem_t<NUM_VARS, NUM_CONSTRAINTS, NUM_NONZERO> &problem,
  const int n_blocks, const int n_threads, const int n_outcomes) {
  GPUSolverMemory<NUM_VARS, NUM_CONSTRAINTS, NUM_NONZERO> out;

  // decltype(problem) *cuda_prob = nullptr;
  cuda_error(cudaMalloc((void **)&out.problem, sizeof(problem)));
  cuda_error(cudaMemcpy((void *)out.problem, &problem, sizeof(problem),
                        cudaMemcpyHostToDevice));

  solution_t<NUM_VARS, NUM_CONSTRAINTS> init_sol{
    .index = 0,
    .upper_bound = std::numeric_limits<int32_t>::max(),
    .obj = 0,
    .var = bitset<NUM_VARS*2>::full(true),
    .rhs = problem.rhs,
    .rhs_n = {},
  };

  // decltype(init_sol) *queue = nullptr;
  // auto const q_max_size = n_blocks * decltype(problem)::n_var;
  // std::vector<decltype(init_sol)> cpu_queue(q_max_size);
  out.queue_max_size = n_blocks * NUM_VARS;
  cuda_error(cudaMalloc((void **)&out.queue, sizeof(init_sol) * out.queue_max_size));
  cuda_error(cudaMemcpy((void *)out.queue, &init_sol, sizeof(init_sol),
                        cudaMemcpyHostToDevice));
  out.queue_size = 1;
  
  // decltype(init_sol) *best_solution = nullptr;
  {
    cuda_error(cudaMalloc((void **)&out.best_solution, sizeof(init_sol)));
    auto cpu_best_sol = init_sol;
    cpu_best_sol.obj = std::numeric_limits<int>::max();
    cuda_error(cudaMemcpy(out.best_solution, &cpu_best_sol, sizeof(init_sol),
                          cudaMemcpyHostToDevice));
  }
  
  // decltype(init_sol) *delta_queue = nullptr;
  out.delta_queue_size = n_blocks * n_outcomes;
  cuda_error(cudaMalloc((void **)&out.delta_queue,
                        sizeof(init_sol) * out.delta_queue_size));
  
  // uint32_t *delta_mask = nullptr;
  // uint32_t *delta_cumsum = nullptr;
  cuda_error(cudaMalloc((void **)&out.delta_mask,
                        sizeof(uint32_t) * out.delta_queue_size));
  
  cuda_error(cudaMalloc((void **)&out.delta_cumsum,
                        sizeof(uint32_t) * out.delta_queue_size));
  cuda_error(
      cudaMemset(out.delta_mask, 0, sizeof(uint32_t) * out.delta_queue_size));
  cuda_error(
      cudaMemset(out.delta_cumsum, 0, sizeof(uint32_t) * out.delta_queue_size));
  
  out.scan_workspace = nullptr;
  out.scan_workspace_size = 0;
  cub::DeviceScan::InclusiveSum(out.scan_workspace, out.scan_workspace_size, out.delta_mask,
                                out.delta_cumsum, n_blocks * n_outcomes);
  cuda_error(cudaMalloc(&out.scan_workspace, out.scan_workspace_size));

  return out;
};

template <size_t n_blocks, size_t n_threads, size_t n_outcomes,
         int NUM_VARS, int NUM_CONSTRAINTS, int NUM_NONZERO>
solution_t<NUM_VARS, NUM_CONSTRAINTS> search(
    GPUSolverMemory<NUM_VARS, NUM_CONSTRAINTS, NUM_NONZERO> gpu_memory) {
  using Solution = solution_t<NUM_VARS, NUM_CONSTRAINTS>;
  auto &cuda_prob = gpu_memory.problem;
  auto &queue = gpu_memory.queue;
  auto &q_size = gpu_memory.queue_size;
  auto &delta_queue = gpu_memory.delta_queue;
  auto &delta_mask = gpu_memory.delta_mask;
  auto &delta_cumsum = gpu_memory.delta_cumsum;
  auto &best_solution = gpu_memory.best_solution;
  auto &d_temp_storage = gpu_memory.scan_workspace;
  auto &temp_storage_bytes = gpu_memory.scan_workspace_size;

  Solution initial_solution;
  cuda_error(cudaMemcpy(&initial_solution, queue, sizeof(Solution),
                        cudaMemcpyDeviceToHost));

  // fmt::println("launching kernel for problem \n{}", problem);
  // fmt::println("initial solution \n{}", initial_solution);

  // int itermax = 20;
  auto const q_max_size = n_blocks * NUM_VARS;
  std::vector<Solution> cpu_queue(q_max_size);
  std::vector<uint32_t> cpu_delta_mask(n_blocks * n_outcomes);

  int iter = 0;
  while (q_size > 0) {
    const bool DEBUG = false;
    auto const n_blocks_l = std::min(q_size, n_blocks);
    if (DEBUG) {
      fmt::println("-----------------------------\nthere {} jobs in the queue, "
                   "launching {}",
                   q_size, n_blocks_l);
    }
    if (DEBUG) {
      traverse<true, n_threads><<<n_blocks_l, n_threads>>>(
          cuda_prob, queue + q_size - n_blocks_l, delta_mask, delta_queue);
    } else {
      traverse<false, n_threads><<<n_blocks_l, n_threads>>>(
          cuda_prob, queue + q_size - n_blocks_l, delta_mask, delta_queue);
    }
    q_size -= n_blocks_l;
  
    // Delta mask is not currently used. Iterates through the newly queued items,
    // finds the fully assigned items and potentially updates the best solution
    if (DEBUG) {
      update_bounds<true, 1024><<<1, 1024>>>(delta_queue, delta_mask,
                                       n_blocks_l * n_outcomes, best_solution);
    } else {
      update_bounds<false, 1024><<<1, 1024>>>(delta_queue, delta_mask,
                                       n_blocks_l * n_outcomes, best_solution);
    }
  
    if (DEBUG) {
      cpu_delta_mask.resize(n_blocks_l * n_outcomes);
      cuda_error(cudaMemcpy(cpu_delta_mask.data(), delta_mask,
                            sizeof(uint32_t) * n_blocks_l * n_outcomes,
                            cudaMemcpyDeviceToHost));
      fmt::println("delta_mask: {}", fmt::join(cpu_delta_mask, ", "));
    }
    // Not sure if it matters, but it seems like this sum could be done in place
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  delta_mask, delta_cumsum,
                                  n_blocks_l * n_outcomes);
    push_back<n_threads><<<n_blocks_l * n_outcomes, n_threads>>>(
        delta_cumsum, delta_queue, queue + q_size);
    uint32_t q_delta = 0;
    cuda_error(cudaMemcpy(&q_delta, delta_cumsum + n_blocks_l * n_outcomes - 1,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (iter % 10000 == 0 || true) {
      Solution cpu_best_sol;
      cuda_error(cudaMemcpy(&cpu_best_sol, best_solution, sizeof(Solution),
                            cudaMemcpyDeviceToHost));
      fmt::println("Iter {} queue_size: {} blocks_launched: {} q_delta: {} best sol obj: {}",
          iter, q_size, n_blocks_l, q_delta, cpu_best_sol.obj);
    }
    q_size += q_delta;

    if (DEBUG){
      fmt::println("q_delta: {}", q_delta);

      cuda_error(cudaMemcpy(cpu_queue.data(), queue, sizeof(Solution) * q_size,
                            cudaMemcpyDeviceToHost));
      for (int i = 0; i < q_size; i++) {
        fmt::println("queue[{}]: \n{}", i, cpu_queue[i]);
      }
      {
        Solution cpu_best_sol;
        cuda_error(cudaMemcpy(&cpu_best_sol, best_solution, sizeof(Solution),
                              cudaMemcpyDeviceToHost));
        fmt::println("best solution: \n{}", cpu_best_sol);
      }
    }
    // if (--itermax == 0)
    //   break;
    iter++;
  }

  {
    Solution cpu_best_sol;
    cuda_error(cudaMemcpy(&cpu_best_sol, best_solution, sizeof(Solution),
                          cudaMemcpyDeviceToHost));
    fmt::println("best solution: \n{}", cpu_best_sol);
  }

  return {};
}

template <int NUM_VARS, int NUM_CONSTRAINTS, int NUM_NONZERO>
void solve_gpu_impl(const MPSData &mps_data) {
  // Convert the problem
  const auto problem = problem_from_mps<NUM_VARS, NUM_CONSTRAINTS, NUM_NONZERO>(mps_data);
  fmt::println("Problem: {}", mps_data.name);

  // Allocate space
  constexpr auto n_blocks = 1024;
  constexpr auto n_threads = 64;
  constexpr auto n_outcomes = 2;
  auto gpu_memory = allocate_gpu_memory<NUM_VARS, NUM_CONSTRAINTS, NUM_NONZERO>(
      problem, n_blocks, n_threads, n_outcomes);

  // Solve the problem
  search<n_blocks, n_threads, n_outcomes>(gpu_memory);
}

void solve_gpu(const MPSData &mps_data) {
  const int num_vars = mps_data.columns.size();
  const int num_constraints = mps_data.rows.size() - 1; // Remove one for the objective
  const int num_nonzero = std::accumulate(
      mps_data.columns.begin(), mps_data.columns.end(), 0, 
      [](const int accum, const auto &var_and_rows){
        return accum + var_and_rows.second.size();
      }) - num_vars; // Remove to account for the objective


  fmt::println("{} num vars: {} num constraints: {} num nonzero: {}",
      mps_data.name, num_vars, num_constraints, num_nonzero);

  if (num_vars == 9 && num_constraints == 14 && num_nonzero == 54) {
    solve_gpu_impl<9, 14, 54>(mps_data);
  } else if (num_vars == 15 && num_constraints == 37 && num_nonzero == 135) {
    solve_gpu_impl<15, 37, 135>(mps_data);
  } else if (num_vars == 45 && num_constraints == 332 && num_nonzero == 1079) {
    solve_gpu_impl<45, 332, 1079>(mps_data);
  } else if (num_vars == 100 && num_constraints == 1 && num_nonzero == 100) {
    solve_gpu_impl<100, 1, 100>(mps_data);
  } else if (num_vars == 201 && num_constraints == 133 && num_nonzero == 1923) {
    solve_gpu_impl<201, 133, 1923>(mps_data);
  } else {
    fmt::println("Unhandled problem size!");
  }
}

