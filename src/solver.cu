#include <algorithm>
#include <sstream>
#include <type_traits>
#ifdef CLANG
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>

#endif

#include "fmt/format.h"
#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>
#include <cuda/std/array>
#include <cuda/std/tuple>
#include <fmt/base.h>
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
  static bitset<n> constexpr from(cuda::std::array<bool, n> const &arr) { 
    bitset<n> res;
    for (int i = 0; i < cdiv(n, 32u); i++) {
      res.data[i] = 0;
    }
    for (int i = 0; i < n; i++) {
      res.set(i, arr[i]);
    }
    return res;
  }
};
template <typename T, int x, int y> struct range_array {
  cuda::std::array<int, x + 1> idx;
  cuda::std::array<T, y> val;

  auto static from_tiplet(
      cuda::std::array<cuda::std::tuple<int, T>, y> const &triplet) {
    cuda::std::array<int, x + 1> delta;
    range_array res;

    for (auto &i : delta)
      i = 0;

    for (auto &i : res.idx)
      i = 0;

    for (auto &[i, _] : triplet)
      ++res.idx[i + 1];

    for (int i = 0; i < x; i++)
      res.idx[i + 1] += res.idx[i];

    for (auto &[i, j] : triplet) {
      res.val[res.idx[i] + delta[i]] = j;
      ++delta[i];
    }
    for (int i = 0; i < x; ++i)
      if (res.idx[i] + delta[i] != res.idx[i + 1]) {
        fmt::println("delta: {}", fmt::join(delta, ", "));
        fmt::println("idx: {}", fmt::join(res.idx, ", "));
        fmt::println("triplet: {}", fmt::join(triplet, ", "));

        throw std::runtime_error("range array construction failed");
      }

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
  cuda::std::array<int, m> rhs_n;
  bitset<m> is_eq;
  // first item is constraint index, second item is coefficient
  range_array<pair<int>, n, n_terms> var_2_constr;
  // first item is variable index, second item is coefficient
  range_array<pair<int>, m, n_terms> constr_2_var;
};

template <int n, int m, int n_terms>
struct fmt::formatter<problem_t<n, m, n_terms>> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  auto format(problem_t<n, m, n_terms> const &t, format_context &ctx) const
      -> format_context::iterator {
    cuda::std::array<cuda::std::array<int, n>, m> A;
    for (auto &c : A) {
      for (auto &x : c) {
        x = 0;
      }
    }
    for (int var_idx = 0; var_idx < n; var_idx++) {
      for (int j = t.var_2_constr.idx[var_idx];
           j < t.var_2_constr.idx[var_idx + 1]; j++) {
        auto const [constr_idx, coeff] = t.var_2_constr.val[j];
        if (A[constr_idx][var_idx] != 0)
          throw std::runtime_error("duplicate variable");
        A[constr_idx][var_idx] = coeff;
      }
    }

    std::stringstream ss;
    ss << fmt::format("problem_t<{}, {}, {}>\n", n, m, n_terms);
    ss << "min ";
    for (int i = 0; i < n; i++) {
      ss << t.obj[i] << "x_" << i;
      if (i != n - 1)
        ss << " + ";
    }

    ss << "\n";

    for (int constr_idx = 0; constr_idx < m; constr_idx++) {
      bool not_first = false;
      for (int var_idx = 0; var_idx < n; var_idx++) {
        if (A[constr_idx][var_idx] == 0)
          continue;
        if (not_first)
          ss << " + ";
        if (A[constr_idx][var_idx] != 1)
          ss << A[constr_idx][var_idx];
        ss << "x_" << var_idx;
        not_first = true;
      }
      if (t.is_eq.get(constr_idx))
        ss << " = ";
      else
        ss << " <= ";
      ss << t.rhs[constr_idx] << "\n";
    }

    return fmt::format_to(ctx.out(), "{}", ss.str());
  }
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

template <int n, int m> struct fmt::formatter<solution_t<n, m>> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  auto format(solution_t<n, m> const &t, format_context &ctx) const
      -> format_context::iterator {
    std::stringstream ss;
    ss << fmt::format("solution_t<{}, {}>\n", n, m);
    ss << "index: " << t.index << "\n";
    ss << "upper_bound: " << t.upper_bound << "\n";
    ss << "obj: " << t.obj << "\n";
    ss << fmt::format("rhs: {}\n", t.rhs);
    ss << fmt::format("rhs_n: {}\n", t.rhs_n);

    ss << "var: \n";
    for (int i = 0; i < n; i++) {
      bool can_be_zero = t.var.get(i * 2);
      bool can_be_one = t.var.get(i * 2 + 1);
      if (can_be_zero && can_be_one)
        ss << "x_" << i << " = 0, 1";
      else if (can_be_zero)
        ss << "x_" << i << " = 0";
      else if (can_be_one)
        ss << "x_" << i << " = 1";
      else
        ss << "x_" << i << " = inf";
      ss << "\n";
    }
    return fmt::format_to(ctx.out(), "{}", ss.str());
  }
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

template <int n_threads, int n_var, int n_constr, int n_terms>
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

  mycpy(qel, &cur);
  __syncthreads();
  if (cur.index >= problem->n_var) {
    if (threadIdx.x <= 1)
      delta_mask[2 * blockIdx.x + threadIdx.x] = 0;
    return;
  }

  for (auto val : {false, true}) {
    __syncthreads();
    auto const dqidx = 2 * blockIdx.x + val;
    kill_switch = false;
    if (!cur.var.get(cur.index * 2 + val)) {
      if (threadIdx.x == 0)
        delta_mask[dqidx] = 0;

      continue;
    }
    __syncthreads();
    mycpy(&cur, &next);
    __syncthreads();
    if (threadIdx.x == 0)
      next.var.set(cur.index * 2, val == false);
    if (threadIdx.x == 1)
      next.var.set(cur.index * 2 + 1, val == true);
    if (threadIdx.x == 2)
      next.index++;
    __syncthreads();
    auto const var_idx = cur.index;
    auto const var_begin = problem->var_2_constr.idx[var_idx];
    auto const var_end = problem->var_2_constr.idx[var_idx + 1];
    for (uint32_t i = threadIdx.x + var_begin; i < var_end && !kill_switch;
         i += blockDim.x) {
      auto const [constr_idx, coeff] = problem->var_2_constr.val[i];
      next.rhs[constr_idx] -= coeff * val;
      ++next.rhs_n[constr_idx];
      auto const rhs_n = next.rhs_n[constr_idx];
      auto const rhs_exp = problem->rhs_n[constr_idx];
      auto const is_eq = problem->is_eq.get(constr_idx);
      if (rhs_n == rhs_exp) {
        if ((is_eq && next.rhs[constr_idx] != 0) ||
            (!is_eq && next.rhs[constr_idx] > 0)) {
          kill_switch = true;
          break;
        }
      }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      if (kill_switch)
        next.obj = std::numeric_limits<int>::max();
      else
        next.obj += problem->obj[cur.index] * val;
    }
    if (threadIdx.x == 0) {
      printf("blockidx %d, val %d, kill_switch %d dqidx %d\n", blockIdx.x, val,
             kill_switch, dqidx);
    }

    if (threadIdx.x == 0) {
      delta_mask[dqidx] = !kill_switch && (next.index < n_var);
    }
    mycpy(&next, delta_queue + dqidx);

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
  if (threadIdx.x == 0)
    s = x;
  __syncthreads();
  return s;
}
template <uint32_t n_threads, int n_var, int n_constr>
__global__ void
update_bounds(solution_t<n_var, n_constr> const *const delta_queue,
              uint32_t *delta_mask, uint32_t delta_q_size,
              solution_t<n_var, n_constr> *best) {
  uint32_t best_idx = 0;
  int best_val = std::numeric_limits<int>::max();
  for (int i = threadIdx.x; i < delta_q_size; i += blockDim.x) {
    printf("delta_queue[%d].index %d obj %d\n", i, delta_queue[i].index,
           delta_queue[i].obj);
    if (delta_queue[i].index == n_var && delta_queue[i].obj < best_val) {
      best_idx = i;
      best_val = delta_queue[i].obj;
    }
  }
  if (best_idx != 0 || best_val != std::numeric_limits<int>::max())
    printf("threadIdx.x %d, best_idx %d best_val %d\n", threadIdx.x, best_idx,
           best_val);
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
  if (threadIdx.x == 0)
    printf("r_best_val %d, r_best_idx %d\n", r_best_val, r_best_idx);
  if (r_best_val < best->obj) {
    mycpy(&delta_queue[r_best_idx], best);
  }
}
int main() {
  /*
max x + 2y + 3z + 4w
s.t.
x + y = 1
z + w = 1
*/
  cuda::std::array<cuda::std::tuple<int, pair<int>>, 4> const var_2_constr{
      cuda::std::make_tuple(0, pair{0, 1}),
      cuda::std::make_tuple(1, pair{0, 1}),
      cuda::std::make_tuple(2, pair{1, 1}),
      cuda::std::make_tuple(3, pair{1, 1})};

  cuda::std::array<cuda::std::tuple<int, pair<int>>, 4> const constr_2_var{
      cuda::std::make_tuple(0, pair{0, 1}),
      cuda::std::make_tuple(0, pair{1, 1}),
      cuda::std::make_tuple(1, pair{2, 1}),
      cuda::std::make_tuple(1, pair{3, 1})};
  auto const problem = problem_t<4, 2, 4>{
      .obj = {1, 2, 3, 4},
      .rhs = {1, 1},
      .rhs_n = {2, 2},
      .is_eq = bitset<2>::full(true),
      .var_2_constr = range_array<pair<int>, 4, 4>::from_tiplet(var_2_constr),
      .constr_2_var = range_array<pair<int>, 2, 4>::from_tiplet(constr_2_var),
  };

  decltype(problem) *cuda_prob = nullptr;
  cuda_error(cudaMalloc((void **)&cuda_prob, sizeof(problem)));
  cuda_error(cudaMemcpy((void *)cuda_prob, &problem, sizeof(problem),
                        cudaMemcpyHostToDevice));
  solution_t<4, 2> init_sol{.index = 0,
                            .upper_bound = std::numeric_limits<int32_t>::max(),
                            .obj = 0,
                            .var = bitset<8>::full(true),
                            .rhs = problem.rhs,
                            .rhs_n = {0, 0}};
  constexpr auto n_blocks = 1024;
  constexpr auto n_threads = 64;
  constexpr auto n_outcomes = 2;
  decltype(init_sol) *queue = nullptr;
  auto const q_max_size = n_blocks * decltype(problem)::n_var;
  std::vector<decltype(init_sol)> cpu_queue(q_max_size);
  cuda_error(cudaMalloc((void **)&queue, sizeof(init_sol) * q_max_size));
  cuda_error(cudaMemcpy((void *)queue, &init_sol, sizeof(init_sol),
                        cudaMemcpyHostToDevice));

  decltype(init_sol) *best_solution = nullptr;
  {
    cuda_error(cudaMalloc((void **)&best_solution, sizeof(init_sol)));
    auto cpu_best_sol = init_sol;
    cpu_best_sol.obj = std::numeric_limits<int>::max();
    cuda_error(cudaMemcpy(best_solution, &cpu_best_sol, sizeof(init_sol),
                          cudaMemcpyHostToDevice));
  }

  decltype(init_sol) *delta_queue = nullptr;
  cuda_error(cudaMalloc((void **)&delta_queue,
                        sizeof(init_sol) * n_blocks * n_outcomes));

  uint32_t *delta_mask = nullptr;
  uint32_t *delta_cumsum = nullptr;
  cuda_error(cudaMalloc((void **)&delta_mask,
                        sizeof(uint32_t) * n_blocks * n_outcomes));

  cuda_error(cudaMalloc((void **)&delta_cumsum,
                        sizeof(uint32_t) * n_blocks * n_outcomes));
  cuda_error(
      cudaMemset(delta_mask, 0, sizeof(uint32_t) * n_blocks * n_outcomes));
  cuda_error(
      cudaMemset(delta_cumsum, 0, sizeof(uint32_t) * n_blocks * n_outcomes));
  auto q_size = 1;

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, delta_mask,
                                delta_cumsum, n_blocks * n_outcomes);
  cuda_error(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  fmt::println("launching kernel for problem \n{}", problem);
  fmt::println("initial solution \n{}", init_sol);
  int itermax = 20;
  std::vector<uint32_t> cpu_delta_mask(n_blocks * n_outcomes);
  while (q_size > 0) {
    auto const n_blocks_l = std::min(q_size, n_blocks);
    fmt::println("-----------------------------\nthere {} jobs in the queue, "
                 "launching {}",
                 q_size, n_blocks_l);
    traverse<n_threads><<<n_blocks_l, n_threads>>>(
        cuda_prob, queue + q_size - n_blocks_l, delta_mask, delta_queue);
    q_size -= n_blocks_l;

    update_bounds<1024><<<1, 1024>>>(delta_queue, delta_mask,
                                     n_blocks_l * n_outcomes, best_solution);

    cpu_delta_mask.resize(n_blocks_l * n_outcomes);
    cuda_error(cudaMemcpy(cpu_delta_mask.data(), delta_mask,
                          sizeof(uint32_t) * n_blocks_l * n_outcomes,
                          cudaMemcpyDeviceToHost));
    fmt::println("delta_mask: {}", fmt::join(cpu_delta_mask, ", "));
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  delta_mask, delta_cumsum,
                                  n_blocks_l * n_outcomes);
    push_back<n_threads><<<n_blocks_l * n_outcomes, n_threads>>>(
        delta_cumsum, delta_queue, queue);
    uint32_t q_detla = 0;
    cuda_error(cudaMemcpy(&q_detla, delta_cumsum + n_blocks_l * n_outcomes - 1,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
    fmt::println("q_delta: {}", q_detla);
    q_size += q_detla;
    cuda_error(cudaMemcpy(cpu_queue.data(), queue, sizeof(init_sol) * q_size,
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < q_size; i++) {
      fmt::println("queue[{}]: \n{}", i, cpu_queue[i]);
    }
    {
      decltype(init_sol) cpu_best_sol;
      cuda_error(cudaMemcpy(&cpu_best_sol, best_solution, sizeof(init_sol),
                            cudaMemcpyDeviceToHost));
      fmt::println("best solution: \n{}", cpu_best_sol);
    }
    if (--itermax == 0)
      break;
  }
};
