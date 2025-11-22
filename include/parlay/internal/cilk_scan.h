// #if defined(CILK_ENABLED) && !defined(CLANGD) && !defined(__CUDACC__)
// #include <cilk/cilk.h>
// #else
// #define cilk_for for
// #define cilk_scope
// #define cilk_spawn
// #define cilk_reducer(id, merge)
// #endif

#include <cilk/reducer>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>


// Reserved to convert non trivially copyable types like pairs
template <class T>
struct is_std_pair : std::false_type {};

template <class T, class U>
struct is_std_pair<std::pair<T, U>> : std::true_type {};

template <class T>
inline constexpr bool is_std_pair_v =
    is_std_pair<std::remove_cv_t<std::remove_reference_t<T>>>::value;

template <class T> struct trivial_pair_like { using type = T; };
template <class A, class B>
struct trivial_pair_like<std::pair<A, B>> {
  struct type {
    A first;
    B second;
  };
};

template <class T>
using trivial_pair_like_t = typename trivial_pair_like<T>::type;

// NOLINTBEGIN(*-reserved-identifier, *-c-arrays, *-owning-memory, *-reinterpret-cast)

// NOLINTBEGIN(*-use-using)
// typedef void (*__cilk_identity_fn)(void *);
// typedef void (*__cilk_reduce_fn)(void *, void *);

typedef void (*__cilk_identity_fn_with_ctx)(void *, void *);
typedef void (*__cilk_reduce_fn_with_ctx)(void *, void *, void *);
// NOLINTEND(*-use-using)

struct range_t {
    int32_t start = 0;
    int32_t end = 0;
};

template <typename V> void id_default(void *v) {
    static_assert(std::is_default_constructible_v<V>,
                  "id_default only works with default constructible types");
    *reinterpret_cast<V *>(v) = V{};
}

template <typename V> void reduce_default(void *l, void *r) {
    auto *lhs = reinterpret_cast<V *>(l);
    auto *rhs = reinterpret_cast<V *>(r);
    *lhs += *rhs;
}

namespace details {

// For now, limit ourselves to containers backed by contiguous memory.
// OpenCilk scanners can actually do arbitrary associative containers where
// entries have stable addresses. But adapting GPU code to support this is a
// huge pain in the ass.

template <typename T> auto get_data_ptr(T &);

template <typename V> auto get_data_ptr(V *&v) -> V * { return v; }

template <typename V> auto get_data_ptr(std::unique_ptr<V[]> &v) -> V * { return v.get(); }

template <typename V> auto get_data_ptr(std::vector<V> &v) -> V * { return v.data(); }

template <typename V> auto get_data_ptr(std::span<V> &v) -> V * { return v.data(); }

template <typename T>
using elem_t = std::remove_pointer_t<decltype(details::get_data_ptr(std::declval<T &>()))>;

} // namespace details

/*
namespace cilk {
template <typename T, __cilk_identity_fn value_id, __cilk_reduce_fn value_reduce>
struct scan_reducer {
    using V = details::elem_t<T>;
    static_assert(std::is_trivially_copyable_v<V>,
                  "scan_reducer only works with trivially copyable types");

    V *data = nullptr;
    V sum;

    // TODO: With compiler support, I think the range can be maintained
    // automatically.
    range_t r{.start = -1, .end = 1};

    // These fields maintain the tree structure for the down-sweep phase.
    scan_reducer *l_child = nullptr, *r_child = nullptr;
    bool is_leftmost = true;

    __attribute__((always_inline)) inline static void value_reduce_to_right(void *left,
                                                                            void *right) {
        V temp = *reinterpret_cast<V *>(left);
        value_reduce(&temp, right);
        *reinterpret_cast<V *>(right) = std::move(temp);
    }

    // Helper routine to perform down-sweep.
    void down_sweep(V &prefix) {
        if (!l_child && !r_child) {
            // At a leaf, broadcast the prefix over the range of the array.
            cilk_for (size_t i = r.start; i < r.end; ++i) {
                value_reduce_to_right(&prefix, &data[i]);
            }
        } else {
            cilk_scope {
                // Both l_child and r_child should be non-null.
                // Add the prefix to the end of l_child's range, to compute the
                // prefix for r_child.
                value_reduce_to_right(&prefix, &data[l_child->r.end]);
                // Recursively down-sweep l_child and r_child in parallel.
                cilk_spawn l_child->down_sweep(prefix);
                r_child->down_sweep(data[l_child->r.end]);
            }
            delete l_child;
            delete r_child;
        }
    }

    static void identity(void *v) {
        auto *sr = new (v) scan_reducer();
        value_id(&sr->sum);
        sr->r.start = -1;
        sr->r.end = -1;

        // No view created by the identity function is leftmost.
        sr->is_leftmost = false;

        // TODO: Need to populate the view's array field.
        // This array should match the argument to __hyper_lookup, so compiler
        // and runtime support could populate this field automatically.
    }

    static void reduce(void *l, void *r) {
        auto *lsr = static_cast<scan_reducer *>(l);
        auto *rsr = static_cast<scan_reducer *>(r);

        // printf("Reducing [%d, %d] and [%d, %d]\n", lsr->r.start, lsr->r.end,
        //        rsr->r.start, rsr->r.end);
        // Perform up-sweep.
        V *data = lsr->data;
        value_reduce_to_right(&data[lsr->r.end], &data[rsr->r.end]);
        value_reduce(&lsr->sum, &rsr->sum);
        if (lsr->is_leftmost) {
            // Only trigger down-sweep when reducing with the leftmost view.
            // Otherwise this hyperobject does too much total work.
            rsr->down_sweep(data[lsr->r.end]);
        } else {
            // Create a tree node with lsr and rsr as children.

            // TODO: We create new scan_reducer views here to avoid problems
            // with the runtime system freeing views implicitly.  Find a better
            // solution than allocating new nodes here.
            auto *l_node = new scan_reducer(*lsr);
            auto *r_node = new scan_reducer(*rsr);
            lsr->l_child = l_node;
            lsr->r_child = r_node;
        }
        // The resulting left view covers the full range.
        lsr->r.end = rsr->r.end;
    }

    scan_reducer() = default;
    explicit scan_reducer(T &array) : data{details::get_data_ptr(array)} { value_id(&sum); }

    struct view_proxy {
        size_t idx;
        scan_reducer<T, value_id, value_reduce> *sr;

        view_proxy(size_t idx, scan_reducer<T, value_id, value_reduce> *sr) : idx(idx), sr(sr) {}
        ~view_proxy() {
            sr->data[idx] = sr->sum;
            sr->r.end = idx;
        }
        view_proxy(const view_proxy &) = delete;
        view_proxy(view_proxy &&other) = delete;
        auto operator=(const view_proxy &) -> view_proxy & = delete;
        auto operator=(view_proxy &&other) -> view_proxy & = delete;
        auto operator*() -> V & { return sr->sum; }
        auto operator->() -> V * { return &sr->sum; }
    };

    auto view(T &array, size_t idx) -> view_proxy {
        if (r.start == -1) {
            this->data = details::get_data_ptr(array);
            this->r.start = static_cast<int32_t>(idx);
            this->r.end = static_cast<int32_t>(idx);
        }
        return {idx, this};
    }
};

template <typename T, __cilk_identity_fn value_id = id_default<details::elem_t<T>>,
          __cilk_reduce_fn value_reduce = reduce_default<details::elem_t<T>>>
// What a hack, this is in case cilk_reducer is a function-like macro
#define SCAN_COMMA ,
using scanner = scan_reducer<T, value_id, value_reduce>
    cilk_reducer(scan_reducer<T SCAN_COMMA value_id SCAN_COMMA value_reduce>::identity,
                 scan_reducer<T SCAN_COMMA value_id SCAN_COMMA value_reduce>::reduce);
#undef SCAN_COMMA
} // namespace cilk
*/

namespace kitcuda {

// This function does not actually exist, it's a marker that the compiler will
// replace with alloca instruction
// extern "C" auto __kitcuda_get_scan_view(size_t size, int32_t *aggregate, int32_t *inclusive_prefix,
//                                         int32_t *scan_state, int32_t *result,
//                                         __cilk_identity_fn identity,
//                                         __cilk_reduce_fn reduce) noexcept -> void *;

extern "C" auto __kitcuda_get_scan_view(size_t size, int32_t *aggregate, int32_t *inclusive_prefix,
                                        int32_t *scan_state, int32_t *result,
                                        __cilk_identity_fn_with_ctx identity, void *identity_ctx,
                                        __cilk_reduce_fn_with_ctx reduce, void *reduce_ctx) noexcept -> void *;

// Effectively nullptr, but blocks LLVM's constant propagation
extern "C" auto __kitcuda_null() noexcept -> void *;

// template <typename T, __cilk_identity_fn value_id = id_default<details::elem_t<T>>,
//           __cilk_reduce_fn value_reduce = reduce_default<details::elem_t<T>>>
template <typename T>
struct scanner {
    using V = details::elem_t<T>;
    static_assert(std::is_trivially_copyable_v<V>,
                  "scanner only works with trivially copyable types");
    static_assert(sizeof(V) % 4 == 0,
                  "scanner only works with types that are multiples of 4 bytes");

    V *data;   // input
    V *result; // output destination
    V *aggregate;
    V *inclusive_prefix;
    int32_t *scan_state;
    __cilk_identity_fn_with_ctx value_id;
    __cilk_reduce_fn_with_ctx value_reduce;
    void *identity_ctx;
    void *reduce_ctx;

    explicit scanner(T &array, __cilk_identity_fn_with_ctx identity, void *identity_ctx_, __cilk_reduce_fn_with_ctx reduce, void *reduce_ctx_)
        : data{details::get_data_ptr(array)},
          result{details::get_data_ptr(array)},
          // If these variables weren't initialized this way, LLVM figures out
          // they are null and const propagates all the way into the kernel, so
          // they won't be among the outlined parameters. We want them to be,
          // because then we can replace them with stuff allocated at runtime in
          // the kernel launch prologue.
          // The interdependencies of hacks is truly mind-boggling.
          aggregate{reinterpret_cast<V *>(__kitcuda_null())},
          inclusive_prefix{reinterpret_cast<V *>(__kitcuda_null())},
          scan_state{reinterpret_cast<int32_t *>(__kitcuda_null())},
          value_id{identity},
          value_reduce{reduce},
          identity_ctx{identity_ctx_},
          reduce_ctx{reduce_ctx_} {}

    // template <typename OutT>
    // scanner(T &input_array, OutT &output_array, __cilk_identity_fn_with_ctx identity, void *identity_ctx_,
    //         __cilk_reduce_fn_with_ctx reduce, void *reduce_ctx_)
    //     : data{details::get_data_ptr(input_array)},
    //       result{details::get_data_ptr(output_array)},
    //       aggregate{reinterpret_cast<V *>(__kitcuda_null())},
    //       inclusive_prefix{reinterpret_cast<V *>(__kitcuda_null())},
    //       scan_state{reinterpret_cast<int32_t *>(__kitcuda_null())},
    //       value_id{identity},
    //       value_reduce{reduce},
    //       identity_ctx{identity_ctx_},
    //       reduce_ctx{reduce_ctx_} {
    // }

    auto view(T &_array, size_t idx) -> V * {
        return reinterpret_cast<V *>(__kitcuda_get_scan_view(
            sizeof(V), reinterpret_cast<int32_t *>(aggregate),
            reinterpret_cast<int32_t *>(inclusive_prefix), reinterpret_cast<int32_t *>(scan_state),
            reinterpret_cast<int32_t *>(result), value_id, identity_ctx, value_reduce, reduce_ctx));
    }
};
} // namespace kitcuda

// NOLINTEND(*-reserved-identifier, *-c-arrays, *-owning-memory, *-reinterpret-cast)
