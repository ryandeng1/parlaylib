
#ifndef PARLAY_SEQUENCE_OPS_H_
#define PARLAY_SEQUENCE_OPS_H_

#include <cmath>
#include <cstddef>

#include <algorithm>
#include <type_traits>
#include <utility>

#include "cilk_scan.h"

#include "../delayed_sequence.h"
#include "../monoid.h"
#include "../parallel.h"
#include "../range.h"
#include "../sequence.h"
#include "../slice.h"
#include "../utilities.h"

#include <cilk/reducer>

template <class Fn>
__cilk_manual_lambda __cilk_make_manual_lambda_id(Fn &&fn) {
  using Stored = std::decay_t<Fn>;
  // auto buffer = __kitcuda_mem_alloc_managed(sizeof(Stored));
  // auto *ctx = new (buffer) Stored(std::forward<Fn>(fn));
  auto *ctx = new Stored(std::forward<Fn>(fn));
  // std::cout << "id sizeof stored: " << sizeof(Stored) << std::endl;
  assert(ctx != nullptr && ctx != NULL);
  return {
    ctx,
    +[](void *c, void *arg) {
      (*static_cast<Stored*>(c))(arg);
    },
    nullptr,
    +[](void *c) { 
        delete static_cast<Stored*>(c);
        // __kitcuda_mem_free(c);
    }
  };
}

template <class Fn>
__cilk_manual_lambda __cilk_make_manual_lambda_reduce(Fn &&fn) {
  using Stored = std::decay_t<Fn>;
  // auto buffer = __kitcuda_mem_alloc_managed(sizeof(Stored));
  // auto *ctx = new (buffer) Stored(std::forward<Fn>(fn));
  auto *ctx = new Stored(std::forward<Fn>(fn));
  assert(ctx != nullptr && ctx != NULL);
  return {
    ctx,
    nullptr,
    +[](void *c, void *lhs, void *rhs) {
      (*static_cast<Stored*>(c))(lhs, rhs);
    },
    +[](void *c) { 
        // __kitcuda_mem_free(c);
        delete static_cast<Stored*>(c);
    }
  };
}

namespace parlay {
namespace internal {

// Return a sequence consisting of the elements
//   f(0), f(1), ... f(n-1)
template<typename UnaryOp>
auto tabulate(size_t n, UnaryOp&& f, size_t granularity = 0) {
  static_assert(std::is_invocable_v<UnaryOp, size_t>);
  static_assert(!std::is_void_v<std::invoke_result_t<UnaryOp, size_t>>);
  static_assert(!std::is_array_v<std::invoke_result_t<UnaryOp, size_t>>);
  return sequence<typename std::decay_t<std::invoke_result_t<UnaryOp, size_t>>>
    ::from_function(n, std::forward<UnaryOp>(f), granularity);
}

// Return a sequence consisting of the elements
//   f(0), f(1), ... f(n-1)
template<typename T, typename UnaryOp>
auto tabulate(size_t n, UnaryOp&& f, size_t granularity = 0) {
  static_assert(std::is_invocable_v<UnaryOp, size_t>);
  static_assert(!std::is_void_v<std::invoke_result_t<UnaryOp, size_t>>);
  static_assert(!std::is_array_v<std::invoke_result_t<UnaryOp, size_t>>);
  static_assert(std::is_convertible_v<std::invoke_result_t<UnaryOp, size_t>, T>);
  return sequence<T>::from_function(n, std::forward<UnaryOp>(f), granularity);
}

// Return a sequence consisting of the elements
//   f(r[0]), f(r[1]), ..., f(r[n-1])
// where n is the size of r.
template<typename R, typename UnaryOp>
auto map(R&& r, UnaryOp&& f, size_t granularity=0) {
  static_assert(is_random_access_range_v<R>);
  static_assert(std::is_invocable_v<UnaryOp, range_reference_type_t<R>>);
  static_assert(!std::is_void_v<std::invoke_result_t<UnaryOp, range_reference_type_t<R>>>);
  return tabulate(parlay::size(r), [f = std::forward<UnaryOp>(f), it = std::begin(r)]
      (size_t i) -> decltype(auto) { return f(it[i]); }, granularity);
}

// Return a delayed sequence consisting of the elements
//   f(0), f(1), ... f(n-1)
template<typename F>
auto delayed_tabulate(size_t n, F f) {
  static_assert(std::is_invocable_v<const F&, size_t>);
  using T = std::invoke_result_t<const F&, size_t>;
  static_assert(!std::is_void_v<T>);
  using V = std::remove_cv_t<std::remove_reference_t<T>>;
  return delayed_sequence<T, V, F>(n, std::move(f));
}

// Return a delayed sequence consisting of the elements
//   f(0), f(1), ... f(n-1)
template<typename T, typename F>
auto delayed_tabulate(size_t n, F f) {
  static_assert(std::is_invocable_v<const F&, size_t>);
  static_assert(std::is_convertible_v<std::invoke_result_t<const F&, size_t>, T>);
  return delayed_sequence<T, std::remove_cv_t<std::remove_reference_t<T>>, F>(n, std::move(f));
}

// Return a delayed sequence consisting of the elements
//   f(0), f(1), ... f(n-1)
template<typename T, typename V, typename F>
auto delayed_tabulate(size_t n, F f) {
  static_assert(std::is_invocable_v<const F&, size_t>);
  static_assert(std::is_convertible_v<std::invoke_result_t<const F&, size_t>, T>);
  return delayed_sequence<T, V, F>(n, std::move(f));
}

// Return a delayed sequence consisting of the elements
//   f(r[0]), f(r[1]), ..., f(r[n-1])
// where n is the size of r.
//
// If r is a temporary, the delayed sequence will take
// ownership of it by moving it. If r is a reference,
// the delayed sequence will hold a reference to it, so
// r must remain alive as long as the delayed sequence.
template<typename R, typename UnaryOp,
    std::enable_if_t<std::is_rvalue_reference_v<R&&>, int> = 0>
auto delayed_map(R&& r, UnaryOp f) {
  static_assert(is_random_access_range_v<R>);
  static_assert(std::is_invocable_v<const UnaryOp&, range_reference_type_t<R>>);
  static_assert(!std::is_void_v<std::invoke_result_t<const UnaryOp&, range_reference_type_t<R>>>);

  // The closure object keeps a cache of the begin iterator to the range r.
  //
  // This is useful because some ranges might perform something like small-size optimization
  // where calling std::begin(r) could have overhead. By keeping the iterator cached, we can
  // random-access from it in the lambda without experiencing this overhead every time
  struct closure {
    closure(R&& r_) : r(std::move(r_)), it(std::begin(r)) {}
    closure(const closure& other) : r(other.r), it(std::begin(r)) {}
    closure(closure&& other) noexcept(std::is_nothrow_move_constructible_v<R>)
        : r(std::move(other.r)), it(std::begin(r)) {}
    closure& operator=(const closure& other) { r = other.r; it = std::begin(r); return *this; }
    closure& operator=(closure&& other) noexcept(std::is_nothrow_move_assignable_v<R>) {
      r = std::move(other.r); it = std::begin(r); return *this; }
    R r;
    range_iterator_type_t<R> it;
  };

  size_t n = parlay::size(r);
  return delayed_tabulate(n, [ c = closure(std::forward<R>(r)), f = std::move(f) ]
      (size_t i) -> decltype(auto) { return f(c.it[i]); });
}

template<typename R, typename UnaryOp,
    std::enable_if_t<std::is_lvalue_reference_v<R&&>, int> = 0>
auto delayed_map(R&& r, UnaryOp f) {
  static_assert(is_random_access_range_v<R>);
  static_assert(std::is_invocable_v<const UnaryOp&, range_reference_type_t<R>>);
  static_assert(!std::is_void_v<std::invoke_result_t<const UnaryOp&, range_reference_type_t<R>>>);

  size_t n = parlay::size(r);
  return delayed_tabulate(n, [ri = std::begin(r), f = std::move(f) ]
      (size_t i) -> decltype(auto) { return f(ri[i]); });
}

// Renamed. Use delayed_tabulate
template <typename F>
auto dseq (size_t n, F f) {
  return delayed_tabulate(n, std::move(f));
}

// Renamed. Use delayed_map.
template<typename R, typename UnaryOp>
auto dmap(R&& r, UnaryOp f) {
  return delayed_map(std::forward<R>(r), std::move(f));
}

template <typename T>
auto singleton(T const &v) -> sequence<T> {
  return sequence<T>(1, v);
}

template <typename Seq, typename Range>
auto copy(Seq const &A, Range R, flags) -> void {
  parallel_for(0, A.size(), [&](size_t i) { R[i] = A[i]; });
}

constexpr const size_t _log_block_size = 10;
constexpr const size_t _block_size = (1 << _log_block_size);

inline size_t num_blocks(size_t n, size_t block_size) {
  if (n == 0)
    return 0;
  else
    return (1 + ((n)-1) / (block_size));
}

template <typename F>
void sliced_for(size_t n, size_t block_size, const F &f, flags fl = no_flag) {
  size_t l = num_blocks(n, block_size);
  auto body = [&](size_t i) {
    size_t s = i * block_size;
    size_t e = (std::min)(s + block_size, n);
    f(i, s, e);
  };
  parallel_for(0, l, body, 1, 0 != (fl & fl_conservative));
}

template <typename Seq, typename Monoid>
auto reduce_serial(Seq const &A, Monoid&& m) {
  static_assert(is_random_access_range_v<Seq>);
  static_assert(is_monoid_for_v<Monoid, range_reference_type_t<Seq>>);
  using T = monoid_value_type_t<Monoid>;
  if (A.size() == 0) return m.identity;
  T r = A[0];
  for (size_t j = 1; j < A.size(); j++) {
    r = m(std::move(r), A[j]);
  }
  return r;
}

// namespace parlay {
// Base case: T is not a sequence
template <typename T>
struct is_parlay_sequence : std::false_type {};

// Specialization: T is a parlay::sequence
// We match the 3 template arguments defined in your file: T, Allocator, EnableSSO
template <typename T, typename Alloc, bool SSO>
struct is_parlay_sequence<parlay::sequence<T, Alloc, SSO>> : std::true_type {};

// Specialization: T is a parlay::slice over contiguous iterators (pointer-like)
template <typename It, typename S>
struct is_parlay_sequence<parlay::slice<It, S>>
    : std::bool_constant<std::is_pointer_v<std::remove_cv_t<It>>> {};

// Helper variable template
// We use std::decay_t to strip const/volatile and references (e.g., sequence& -> sequence)
template <typename T>
inline constexpr bool is_parlay_sequence_v = is_parlay_sequence<std::decay_t<T>>::value;
// }

template <typename Seq, typename Monoid>
auto reduce(Seq const &A, Monoid&& m, flags fl = no_flag) {
  static_assert(is_random_access_range_v<Seq>);
  static_assert(is_monoid_for_v<Monoid, range_reference_type_t<Seq>>);
  using T = monoid_value_type_t<Monoid>;
  size_t n = A.size();
  size_t block_size = (std::max)(_block_size, 4 * static_cast<size_t>(std::ceil(std::sqrt(n))));
  size_t l = num_blocks(n, block_size);
  if (l == 0) return m.identity;
  if (l == 1 || (fl & fl_sequential)) {
    return reduce_serial(A, m);
  }

  
  // std::function ident_fn = [=](void *v) { new (v) T(m.identity); };
  // std::function reduce_fn = [=](void *l, void *r) {
  //   *static_cast<T *>(l) = m(*static_cast<T *>(l), *static_cast<T *>(r));
  //   if (std::is_destructible<T>::value) static_cast<T *>(r)->~T();
  // };

  // T cilk_reducer(ident_fn, reduce_fn) r = m.identity;

  auto identity = [=](void *v) { new (v) T(m.identity); };
  auto reduce = [=](void *l, void *r) {
    *static_cast<T *>(l) = m(*static_cast<T *>(l), *static_cast<T *>(r));
    // if (std::is_destructible<T>::value) static_cast<T *>(r)->~T();
  };

  auto identity_ = __cilk_make_manual_lambda_id(identity);
  auto reduce_ = __cilk_make_manual_lambda_reduce(reduce);

  auto _Monoid = __reducer_callbacks{
    sizeof(T),
    identity_,
    reduce_,
  };

  T cilk_reducer(_Monoid) r = m.identity;

  if constexpr (is_parlay_sequence_v<Seq>) {
    auto *begin = A.begin();
    [[tapir::target("cuda")]] cilk_for (size_t i = 0; i < n; i++) {
      const auto& x = begin[i];
      // NOTE: Need to explicitly convert the hyperobject back into a view here, to work around type-deduction issues.
      r = m(std::move(*&r), x);
    }

    return *&r;
  } else {
    [[tapir::target("cuda")]] cilk_for (size_t i = 0; i < n; i++) {
      const auto& x = A[i];
      // NOTE: Need to explicitly convert the hyperobject back into a view here, to work around type-deduction issues.
      r = m(std::move(*&r), x);
    }

    // NOTE: Need to explicitly convert the hyperobject back into a view here, to work around type-deduction issues.
    return *&r;
  }

  //// ORIGINAL PARLAYLIB CODE ////
  // auto sums = sequence<T>::uninitialized(l);
  // sliced_for(n, block_size, [&](size_t i, size_t s, size_t e) {
  //   assign_uninitialized(sums[i], reduce_serial(make_slice(A).cut(s, e), m));
  // });
  // T r = internal::reduce(sums, m);
  // return r;
}

const flags fl_scan_inclusive = (1 << 4);

template <typename In_Seq, typename Out_Seq, typename Monoid>
auto scan_serial(In_Seq const &In, Out_Seq Out, Monoid&& m,
                 monoid_value_type_t<Monoid> offset, flags fl, bool out_uninitialized = false) {
  static_assert(is_random_access_range_v<In_Seq>);
  static_assert(is_monoid_for_v<Monoid, range_reference_type_t<In_Seq>>);
  using T = monoid_value_type_t<Monoid>;
  T r = std::move(offset);
  size_t n = In.size();
  bool inclusive = fl & fl_scan_inclusive;
  if (inclusive) {
    for (size_t i = 0; i < n; i++) {
      r = m(std::move(r), In[i]);
      if (out_uninitialized) assign_uninitialized(Out[i], r);
      else Out[i] = r;
    }
  } else {
    for (size_t i = 0; i < n; i++) {
      T t = In[i];
      if (out_uninitialized) assign_uninitialized(Out[i], r);
      else Out[i] = r;
      r = m(std::move(r), t);
    }
  }
  return r;
}

template <bool Alias = false, typename In_Seq, typename Out_Range, class Monoid>
auto scan_(In_Seq const &In, Out_Range Out, Monoid&& m, flags fl, bool out_uninitialized=false) {
  static_assert(is_random_access_range_v<In_Seq>);
  static_assert(is_monoid_for_v<Monoid, range_reference_type_t<In_Seq>>);
  using T = monoid_value_type_t<Monoid>;
  using V = details::elem_t<T *>;

  size_t n = In.size();
  size_t l = num_blocks(n, _block_size);

  if (n == 0) {
    return m.identity;
  }

  if (l <= 2 || fl & fl_sequential) {
    return scan_serial(In, Out, m, m.identity, fl, out_uninitialized);
  }

  if constexpr (!std::is_trivially_copyable_v<V>) {
    return scan_serial(In, Out, m, m.identity, fl, out_uninitialized);
  } else {
    auto identity = __cilk_make_manual_lambda_id([=](void *v) {
      new (v) T(m.identity);
    });

    auto reduce = __cilk_make_manual_lambda_reduce([=](void *l, void *r) {
      *static_cast<T *>(l) = m(*static_cast<T *>(l), *static_cast<T *>(r));
    });

    bool inclusive = fl & fl_scan_inclusive;

    const auto in_ptr = [&]() {
        if constexpr (is_parlay_sequence_v<In_Seq> ) {
            return In.begin();      // Only compiled if T is int
        } else {
            return In;     // Only compiled if T is not int
        }
    }();

    auto out_ptr = [&]() {
        if constexpr (is_parlay_sequence_v<Out_Range> ) {
            return Out.begin();      // Only compiled if T is int
        } else {
            return Out;     // Only compiled if T is not int
        }
    }();

    // bool alias = (in_ptr == out_ptr);
    bool alias = Alias;
    T last_input = in_ptr[n - 1];

    if (inclusive) {
      if (out_uninitialized && !alias) {
        for (size_t i = 0; i < n; i++) {
          assign_uninitialized(out_ptr[i], in_ptr[i]);
        }
      } else {
        for (size_t i = 0; i < n; i++) {
          out_ptr[i] = in_ptr[i];
        }
      }
    } else {
      if (alias) {
        T first_input = in_ptr[0];
        for (size_t i = n; i-- > 1;) {
          out_ptr[i] = in_ptr[i - 1];
        }
        out_ptr[0] = m.identity;
        if (n > 1) out_ptr[1] = first_input;
      } else {
        if (out_uninitialized) {
          assign_uninitialized(out_ptr[0], m.identity);
          for (size_t i = 1; i < n; i++) {
            assign_uninitialized(out_ptr[i], in_ptr[i - 1]);
          }
        } else {
          out_ptr[0] = m.identity;
          for (size_t i = 1; i < n; i++) {
            out_ptr[i] = in_ptr[i - 1];
          }
        }
      }
    }

    kitcuda::scanner<T *> scanner(out_ptr, identity.invoke1, identity.ctx,
                                  reduce.invoke2, reduce.ctx);
    [[tapir::target("cuda")]]
    cilk_for(size_t i = 0; i < n; ++i) {
      auto view = scanner.view(out_ptr, i);
      *view = m(std::move(*view), out_ptr[i]);
    }

    T total = inclusive ? out_ptr[n - 1] : m(out_ptr[n - 1], last_input);
    return total;

    // if constexpr (is_parlay_sequence_v<In_Seq> && is_parlay_sequence_v<Out_Range>) {
    //   // const T *in_ptr = In.begin();
    //   // T *out_ptr = Out.begin();
    //   bool alias = (in_ptr == out_ptr);
    //   T last_input = in_ptr[n - 1];

    //   if (inclusive) {
    //     if (out_uninitialized && !alias) {
    //       for (size_t i = 0; i < n; i++) {
    //         assign_uninitialized(out_ptr[i], in_ptr[i]);
    //       }
    //     } else {
    //       for (size_t i = 0; i < n; i++) {
    //         out_ptr[i] = in_ptr[i];
    //       }
    //     }
    //   } else {
    //     if (alias) {
    //       T first_input = in_ptr[0];
    //       for (size_t i = n; i-- > 1;) {
    //         out_ptr[i] = in_ptr[i - 1];
    //       }
    //       out_ptr[0] = m.identity;
    //       if (n > 1) out_ptr[1] = first_input;
    //     } else {
    //       if (out_uninitialized) {
    //         assign_uninitialized(out_ptr[0], m.identity);
    //         for (size_t i = 1; i < n; i++) {
    //           assign_uninitialized(out_ptr[i], in_ptr[i - 1]);
    //         }
    //       } else {
    //         out_ptr[0] = m.identity;
    //         for (size_t i = 1; i < n; i++) {
    //           out_ptr[i] = in_ptr[i - 1];
    //         }
    //       }
    //     }
    //   }

    //   kitcuda::scanner<T *> scanner(out_ptr, identity.invoke1, identity.ctx,
    //                                 reduce.invoke2, reduce.ctx);
    //   [[tapir::target("cuda")]]
    //   cilk_for(size_t i = 0; i < n; ++i) {
    //     auto view = scanner.view(out_ptr, i);
    //     *view = m(std::move(*view), out_ptr[i]);
    //   }

    //   T total = inclusive ? out_ptr[n - 1] : m(out_ptr[n - 1], last_input);
    //   return total;
    // } else {
    //   // Initialize seq_arr with input values for inclusive scan, or shifted values for exclusive scan
    //   auto seq_arr = new T[n];
    //   if (inclusive) {
    //     for (size_t i = 0; i < n; i++) {
    //       seq_arr[i] = In[i];
    //     }
    //   } else {
    //     seq_arr[0] = m.identity;
    //     for (size_t i = 1; i < n; i++) {
    //       seq_arr[i] = In[i - 1];
    //     }
    //   }

    //   kitcuda::scanner<T *> scanner(seq_arr, identity.invoke1, identity.ctx,
    //                                 reduce.invoke2, reduce.ctx);

    //   [[tapir::target("cuda")]]
    //   cilk_for(size_t i = 0; i < n; ++i) {
    //     auto view = scanner.view(seq_arr, i);
    //     *view = m(std::move(*view), seq_arr[i]);
    //   }

    //   T total = seq_arr[n - 1];

    //   // For exclusive scan, the total should include the last element from input
    //   if (!inclusive) {
    //     total = m(std::move(total), In[n - 1]);
    //   }

    //   // Copy the scanned results back to output
    //   if (out_uninitialized) {
    //     for (size_t i = 0; i < n; i++) {
    //       assign_uninitialized(Out[i], seq_arr[i]);
    //     }
    //   } else {
    //     for (size_t i = 0; i < n; i++) {
    //       Out[i] = seq_arr[i];
    //     }
    //   }

    //   delete[] seq_arr;
    //   return total;
    // }
  }

  /*
  std::function ident_fn = [=](void *v) { new (v) T(m.identity); };
  std::function reduce_fn = [=](T *l, T *r) { return m(*l, *r); };

  auto identity = [=](void *v) { new (v) T(m.identity); };
  auto reduce = [=](T *l, T *r) {
    return m(*l, *r);
  };

  auto identity_ = __cilk_make_manual_lambda_id(identity);
  auto reduce_ = __cilk_make_manual_lambda_reduce_scan<T>(reduce);

  scanner<Out_Range> base(Out, identity_, reduce_, inclusive);

  auto reducer_ctx = scan_detail::reducer_context<Out_Range>{
      .out = &Out,
      .value_id = &identity_,
      .value_reduce = &reduce_,
      .inclusive = inclusive};

  const __reducer_callbacks _Monoid = {
    .size = sizeof(scanner<Out_Range>),
    .identity = scan_detail::make_identity_lambda<Out_Range>(reducer_ctx),
    .reduce = scan_detail::make_reduce_lambda<Out_Range>()
  };

  scanner<Out_Range> cilk_reducer(_Monoid) scanner = base;

  if (inclusive) {
    [[tapir::target("cuda")]] 
    cilk_for(size_t i = 0; i < n; ++i) {
    // for(size_t i = 0; i < n; ++i) {
      auto view = scanner.view(i);
      *view = m(std::move(*&view), In[i]);
    }
  } else {
    [[tapir::target("cuda")]] 
    cilk_for(size_t i = 0; i < n; ++i) {
    // for(size_t i = 0; i < n; ++i) {
      T t = In[i];
      auto view = scanner.view(i);
      *view = m(std::move(*&view), t);
    }
  }
  T total = scanner.sum;
  */

  // //// ORIGINAL PARLAYLIB CODE ////
  // auto sums = sequence<T>::uninitialized(l);
  // sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
  //   assign_uninitialized(sums[i], reduce_serial(make_slice(In).cut(s, e), m));
  // });
  // T total = scan_serial(sums, make_slice(sums), m, m.identity, 0, false);
  // sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
  //   auto O = make_slice(Out).cut(s, e);
  //   scan_serial(make_slice(In).cut(s, e), O, m, sums[i], fl, out_uninitialized);
  // });

  // return total;
}

/*
template <typename In_Seq, typename Out_Range, class Monoid>
auto scan_(In_Seq const &In, Out_Range Out, Monoid&& m, flags fl, bool out_uninitialized=false) {
  static_assert(is_random_access_range_v<In_Seq>);
  static_assert(is_monoid_for_v<Monoid, range_reference_type_t<In_Seq>>);
  using T = monoid_value_type_t<Monoid>;
  size_t n = In.size();
  size_t l = num_blocks(n, _block_size);
  if (l <= 2 || fl & fl_sequential)
    return scan_serial(In, Out, m, m.identity, fl, out_uninitialized);

  // std::function ident_fn = [=](void *v) { new (v) T(m.identity); };
  // std::function reduce_fn = [=](T *l, T *r) { return m(*l, *r); };
  auto ident_fn = [=](void *v) { new (v) T(m.identity); };
  auto reduce_fn = [=](T *l, T *r) { return m(*l, *r); };
  bool inclusive = fl & fl_scan_inclusive;
  // // FIXME: It's awkward that we need to separately create a non-reducer scanner object, so that the reducer object can
  // // refer to the object's identity and reduce methods.
  // scanner<Out_Range> base(Out, ident_fn, reduce_fn, inclusive);
  // // scanner<Out_Range> cilk_reducer(base.identity, base.reduce) scanner = base;
  // const __reducer_callbacks _Monoid = {
  //     .size = sizeof(scanner<Out_Range>), .identity = base.identity, .reduce = base.reduce};
  // scanner<Out_Range> cilk_reducer(_Monoid) scanner = base.identity;
  // scanner<Out_Range> cilk_reducer scanner(Out, ident_fn, reduce_fn, inclusive);
  scanner<Out_Range> base(Out, ident_fn, reduce_fn, inclusive);
  const __reducer_callbacks _Monoid = {
      .size = sizeof(scanner<Out_Range>), .identity = __cilk_make_manual_lambda(ident_fn), .reduce = __cilk_make_manual_lambda(reduce_fn)};
  scanner<Out_Range> cilk_reducer(_Monoid) scanner = base;

  if (inclusive) {
    for(size_t i = 0; i < n; ++i) {
    // for(size_t i = 0; i < n; ++i) {
      auto view = scanner.view(i);
      *view = m(std::move(*&view), In[i]);
    }
  } else {
    for(size_t i = 0; i < n; ++i) {
    // for(size_t i = 0; i < n; ++i) {
      T t = In[i];
      auto view = scanner.view(i);
      *view = m(std::move(*&view), t);
    }
  }
  T total = scanner.sum;

  // //// ORIGINAL PARLAYLIB CODE ////
  // auto sums = sequence<T>::uninitialized(l);
  // sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
  //   assign_uninitialized(sums[i], reduce_serial(make_slice(In).cut(s, e), m));
  // });
  // T total = scan_serial(sums, make_slice(sums), m, m.identity, 0, false);
  // sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
  //   auto O = make_slice(Out).cut(s, e);
  //   scan_serial(make_slice(In).cut(s, e), O, m, sums[i], fl, out_uninitialized);
  // });

  return total;
}
*/

/*
template <typename In_Seq, typename Out_Range, class Monoid>
auto scan_(In_Seq const &In, Out_Range Out, Monoid&& m, flags fl, bool out_uninitialized=false) {
  static_assert(is_random_access_range_v<In_Seq>);
  static_assert(is_monoid_for_v<Monoid, range_reference_type_t<In_Seq>>);
  using T = monoid_value_type_t<Monoid>;
  size_t n = In.size();
  size_t l = num_blocks(n, _block_size);
  if (l <= 2 || fl & fl_sequential)
    return scan_serial(In, Out, m, m.identity, fl, out_uninitialized);
  auto sums = sequence<T>::uninitialized(l);
  sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
    assign_uninitialized(sums[i], reduce_serial(make_slice(In).cut(s, e), m));
  });
  T total = scan_serial(sums, make_slice(sums), m, m.identity, 0, false);
  sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
    auto O = make_slice(Out).cut(s, e);
    scan_serial(make_slice(In).cut(s, e), O, m, sums[i], fl, out_uninitialized);
  });
  return total;
}
*/

template <typename Iterator, typename Monoid>
auto scan_inplace(slice<Iterator, Iterator> In, Monoid&& m, flags fl = no_flag) {
  static_assert(is_monoid_for_v<Monoid, iterator_reference_type_t<Iterator>>);
  return scan_<true>(In, In, std::forward<Monoid>(m), fl);
}

template <typename In_Seq, typename Monoid>
auto scan(In_Seq const &In, Monoid&& m, flags fl = no_flag) {
  static_assert(is_random_access_range_v<In_Seq>);
  static_assert(is_monoid_for_v<Monoid, range_reference_type_t<In_Seq>>);
  using T = monoid_value_type_t<Monoid>;
  auto Out = sequence<T>::uninitialized(In.size());
  return std::make_pair(std::move(Out), scan_<false>(In, make_slice(Out), std::forward<Monoid>(m), fl, true));
}

// do in place if rvalue reference to a sequence<T>
template <typename T, typename Monoid,
    std::enable_if_t<std::is_same_v<T, monoid_value_type_t<Monoid>>, int> = 0>
auto scan(sequence<T>&& In, Monoid&& m, flags fl = no_flag) {
  static_assert(is_monoid_v<Monoid>);
  sequence<T> Out = std::move(In);
  T total = scan_<true>(make_slice(Out), make_slice(Out), std::forward<Monoid>(m), fl);
  return std::make_pair(std::move(Out), total);
}

template <typename Seq>
size_t sum_bools_serial(Seq const &I) {
  size_t r = 0;
  for (size_t j = 0; j < I.size(); j++) {
    r += static_cast<bool>(I[j]);
  }
  return r;
}

template <typename In_Seq, typename Bool_Seq>
auto pack_serial(In_Seq const &In, Bool_Seq const &Fl)
    -> sequence<typename In_Seq::value_type> {
  using T = typename In_Seq::value_type;
  size_t n = In.size();
  size_t m = sum_bools_serial(Fl);
  sequence<T> Out = sequence<T>::uninitialized(m);
  size_t k = 0;
  for (size_t i = 0; i < n; i++)
    if (Fl[i]) assign_uninitialized(Out[k++], In[i]);
  return Out;
}


template <class Slice, class Slice2, typename Out_Seq>
size_t pack_serial_at(Slice In, Slice2 Fl, Out_Seq Out) {
  size_t k = 0;
  for (size_t i = 0; i < In.size(); i++)
    if (Fl[i]) assign_uninitialized(Out[k++], In[i]);
  return k;
}

template <typename In_Seq, typename Bool_Seq>
auto pack(In_Seq const &In, Bool_Seq const &Fl, flags fl = no_flag)
    -> sequence<typename In_Seq::value_type> {
  using T = typename In_Seq::value_type;
  size_t n = In.size();
  size_t l = num_blocks(n, _block_size);
  if (l == 1 || fl & fl_sequential) return pack_serial(In, Fl);
  auto sums = sequence<size_t>::uninitialized(l);
  sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
    assign_uninitialized(sums[i], sum_bools_serial(make_slice(Fl).cut(s, e)));
  });
  size_t m = scan_inplace(make_slice(sums), plus<size_t>());
  sequence<T> Out = sequence<T>::uninitialized(m);
  sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
    pack_serial_at(make_slice(In).cut(s, e), make_slice(Fl).cut(s, e),
                   make_slice(Out).cut(sums[i], (i == l - 1) ? m : sums[i + 1]));
  });
  return Out;
}

// Pack the output to the output range.
template <typename In_Seq, typename Bool_Seq, typename Out_Seq>
size_t pack_out(In_Seq const &In, Bool_Seq const &Fl, /* uninitialized */ Out_Seq Out, flags fl = no_flag) {
  size_t n = In.size();
  size_t l = num_blocks(n, _block_size);
  if (l <= 1 || fl & fl_sequential) {
    return pack_serial_at(In, make_slice(Fl).cut(0, In.size()), Out);
  }
  sequence<size_t> Sums(l);
  sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
    Sums[i] = sum_bools_serial(make_slice(Fl).cut(s, e));
  });
  size_t m = scan_inplace(make_slice(Sums), plus<size_t>());
  sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
    pack_serial_at(make_slice(In).cut(s, e), make_slice(Fl).cut(s, e),
                   make_slice(Out).cut(Sums[i], (i == l - 1) ? m : Sums[i + 1]));
  });
  return m;
}

// like filter but applies g before returning result
template <typename In_Seq, typename F, typename G>
auto filter_map(In_Seq const &In, F&& f, G&& g) {
  using outT = std::invoke_result_t<G, range_reference_type_t<In_Seq>>;
  size_t n = In.size();
  size_t l = num_blocks(n, _block_size);
  // auto in_mapped = delayed_seq<outT>(n, [&] (size_t i) { return g(In[i]); });
  auto in_mapped = delayed_tabulate<outT>(n, [&] (size_t i) { return g(In[i]); });

  sequence<size_t> Sums(l);
  sequence<bool> Fl(n);
  sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
    size_t r = 0;
    for (size_t j = s; j < e; j++) r += (Fl[j] = f(In[j]));
    Sums[i] = r;
  });
  size_t m = scan_inplace(make_slice(Sums), plus<size_t>());
  sequence<outT> Out = sequence<outT>::uninitialized(m);
  sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
    pack_serial_at(make_slice(in_mapped).cut(s, e),
		   make_slice(Fl).cut(s, e), make_slice(Out).cut(Sums[i], (i == l - 1) ? m : Sums[i + 1]));
  });
  return Out;
}

template <typename In_Seq, typename F>
auto filter(In_Seq const &In, F&& f) -> sequence<typename In_Seq::value_type> {
  auto identity = [](auto&& x) -> range_value_type_t<In_Seq> { return std::forward<decltype(x)>(x); };
  return filter_map(In, std::forward<F>(f), identity);
}

template <typename In_Seq, typename F>
auto filter(In_Seq const &In, F&& f, flags) {
  return filter(In, std::forward<F>(f));
}

// Filter and write the output to the output range.
template <typename In_Seq, typename Out_Seq, typename F>
size_t filter_out(In_Seq const &In, /* uninitialized */ Out_Seq Out, F&& f) {
  size_t n = In.size();
  size_t l = num_blocks(n, _block_size);
  sequence<size_t> Sums(l);
  sequence<bool> Fl(n);
  sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
    size_t r = 0;
    for (size_t j = s; j < e; j++) r += (Fl[j] = f(In[j]));
    Sums[i] = r;
  });
  size_t m = scan_inplace(make_slice(Sums), plus<size_t>());
  sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
    pack_serial_at(make_slice(In).cut(s, e), make_slice(Fl).cut(s, e),
                   make_slice(Out).cut(Sums[i], (i == l - 1) ? m : Sums[i + 1]));
  });
  return m;
}

template <typename In_Seq, typename Out_Seq, typename F>
size_t filter_out(In_Seq const &In, /* uninitialized */ Out_Seq Out, F&& f, flags) {
  return filter_out(In, Out, std::forward<F>(f));
}

template <typename Idx_Type, typename Bool_Seq>
auto pack_index(Bool_Seq const &Fl, flags fl = no_flag) {
  auto identity = [](size_t i) -> Idx_Type { return static_cast<Idx_Type>(i); };
  return pack(delayed_tabulate(Fl.size(), identity), Fl, fl);
}

template <typename assignment_tag, typename InIterator, typename OutIterator, typename Char_Seq>
std::pair<size_t, size_t> split_three(slice<InIterator, InIterator> In,
                                      slice<OutIterator, OutIterator> Out,
                                      Char_Seq const &Fl, flags fl = no_flag) {
  size_t n = In.size();
  size_t l = num_blocks(n, _block_size);
  sequence<size_t> Sums0(l);
  sequence<size_t> Sums1(l);
  sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
    size_t c0 = 0;
    size_t c1 = 0;
    for (size_t j = s; j < e; j++) {
      if (Fl[j] == 0) c0++;
      else if (Fl[j] == 1) c1++;
    }
    Sums0[i] = c0;
    Sums1[i] = c1;
  }, fl);
  size_t m0 = scan_inplace(make_slice(Sums0), plus<size_t>());
  size_t m1 = scan_inplace(make_slice(Sums1), plus<size_t>());
  sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
    size_t c0 = Sums0[i];
    size_t c1 = m0 + Sums1[i];
    size_t c2 = m0 + m1 + (s - Sums0[i] - Sums1[i]);
    for (size_t j = s; j < e; j++) {
      if (Fl[j] == 0) {
        assign_dispatch(Out[c0++], In[j], assignment_tag());
      } else if (Fl[j] == 1) {
        assign_dispatch(Out[c1++], In[j], assignment_tag());
      } else {
        assign_dispatch(Out[c2++], In[j], assignment_tag());
      }
    }
  }, fl);
  return std::make_pair(m0, m1);
}

template <typename In_Seq, typename Bool_Seq>
auto split_two(In_Seq const &In, Bool_Seq const &Fl, flags fl = no_flag)
    -> std::pair<sequence<typename In_Seq::value_type>, size_t> {
  using T = typename In_Seq::value_type;
  size_t n = In.size();
  size_t l = num_blocks(n, _block_size);
  sequence<size_t> Sums(l);
  sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
    size_t c = 0;
    for (size_t j = s; j < e; j++) c += (Fl[j] == false);
    Sums[i] = c;
  }, fl);
  size_t m = scan_inplace(make_slice(Sums), plus<size_t>());
  sequence<T> Out = sequence<T>::uninitialized(n);
  sliced_for(n, _block_size, [&](size_t i, size_t s, size_t e) {
    size_t c0 = Sums[i];
    size_t c1 = s + (m - c0);
    for (size_t j = s; j < e; j++) {
      if (Fl[j] == false)
        assign_uninitialized(Out[c0++], In[j]);
      else
        assign_uninitialized(Out[c1++], In[j]);
    }
  }, fl);
  return std::make_pair(std::move(Out), m);
}

}  // namespace internal
}  // namespace parlay

#endif  // PARLAY_SEQUENCE_OPS_H_
