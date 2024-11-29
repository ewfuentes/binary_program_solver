#pragma once

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
      ss << t.obj[i] << " x_" << i;
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
        ss << " x_" << var_idx;
        not_first = true;
      }
      if (t.is_eq.get(constr_idx))
        ss << " = ";
      else
        ss << " <= ";
      ss << t.rhs[constr_idx] << fmt::format(" (num terms: {})", t.rhs_n.at(constr_idx)) << "\n";
    }

    return fmt::format_to(ctx.out(), "{}", ss.str());
  }
};


template <int n, int m> struct fmt::formatter<solution_t<n, m>> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  auto format(solution_t<n, m> const &t, format_context &ctx) const
      -> format_context::iterator {
    std::stringstream ss;
    ss << fmt::format("solution_t<{}, {}>\n", n, m);
    ss << "index: " << t.index << "\n";
    ss << "remaining_lower_bound: " << t.remaining_lower_bound << "\n";
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
      ss << " ";
      // ss << "\n";
    }
    return fmt::format_to(ctx.out(), "{}", ss.str());
  }
};

