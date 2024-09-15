/// @file meta.h
/// @brief This header file contains metaprogramming constructs.

#ifndef UTILS_META_H
#define UTILS_META_H

#include <type_traits>

namespace t8gpu::meta {

  /// @brief all_same is a bool integral constant struct whose value
  ///        static field is true only if all of the types stripped
  ///        from cv supplied are equal.
  ///
  /// @tparam Ts... the types to check.
  ///
  /// Here are a few usage examples:
  ///
  /// all_same<int, int, const int> = std::true_type
  /// all_same<double, float>       = std::false_type
  /// all_same<int, int&>           = std::false_type
  /// all_same<int>                 = std::true_type
  /// all_same<>                    = std::false_type
  ///
  template<typename... Ts>
  struct all_same : std::false_type {};

  template<typename T>
  struct all_same<T> : std::true_type {};

  template<typename T1, typename T2, typename... Ts>
  struct all_same<T1, T2, Ts...> : std::integral_constant<bool,
                                                          (std::is_same_v<std::remove_cv_t<T1>, std::remove_cv_t<T2>> &&
                                                           all_same<T2, Ts...>::value)> {};

  template<typename... Ts>
  inline constexpr bool all_same_v = all_same<Ts...>::value;

  /// @brief is_explicitly_convertible_to is a bool integral constant
  ///        whose value represents whether the type T can be
  ///        explicitly statically casted to U. This is more
  ///        permissive than std::is_convertible which only is true
  ///        for implicit casts.
  ///
  /// @tparam T the type to be converted from.
  /// @tparam U the type to be converted to.
  ///
  template<typename T, typename U, typename = void>
  struct is_explicitly_convertible_to : std::false_type {};

  template<typename T, typename U>
  struct is_explicitly_convertible_to<T, U, std::void_t<decltype(static_cast<U>(std::declval<T>()))>> : std::true_type {
  };

  template<typename T, typename U>
  inline constexpr bool is_explicitly_convertible_to_v = is_explicitly_convertible_to<T, U>::value;

  /// @brief argpack_at returns the pack argument at the given
  ///        index. If the index is not in the range, it does not
  ///        compile.
  ///
  /// @tparam index   the index of the pack argument we wish to retrieve.
  /// @tparam args... the argument pack of ints.
  template<int index, int... args>
  struct argpack_at {};

  template<int arg1>
  struct argpack_at<0, arg1> : std::integral_constant<int, arg1> {};

  template<int index, int arg1, int... args>
  struct argpack_at<index, arg1, args...>
      : std::conditional_t<(index == 0), std::integral_constant<int, arg1>, argpack_at<index - 1, args...>> {};

  template<int... args>
  inline constexpr int argpack_at_v = argpack_at<args...>::value;

  /// @brief Computes the product of pack arguments from an index. If
  ///        the index is greater than the pack size, it returns 1.
  ///
  /// @tparam index   the index from witch we want to start multiplying.
  /// @tparam arg1    the first argument pack integer.
  /// @tparam args... the remainder integer arguments.
  template<int index, int arg1, int... args>
  struct argpack_mul_from
      : std::integral_constant<int,
                               ((index == 0) ? arg1 : 1) *
                                   argpack_mul_from<(index == 0) ? index : index - 1, args...>::value> {};

  template<int index, int x>
  struct argpack_mul_from<index, x> : std::integral_constant<int, (index == 0) ? x : 1> {};

  template<int index, int arg1, int... args>
  inline constexpr int argpack_mul_from_v = argpack_mul_from<index, arg1, args...>::value;

  /// @brief Computes the product of pack arguments up to an index.
  ///
  /// @tparam index   the index from witch we want to stop multiplying.
  /// @tparam arg1    the first argument pack integer.
  /// @tparam args... the remainder integer arguments.
  template<int index, int arg1, int... args>
  struct argpack_mul_to : std::integral_constant<int,
                                                 ((index == 0) ? 1 : arg1) *
                                                     argpack_mul_to<(index == 0) ? index : index - 1, args...>::value> {
  };

  template<int index, int x>
  struct argpack_mul_to<index, x> : std::integral_constant<int, (index == 0) ? 1 : x> {};

  template<int index, int arg1, int... args>
  inline constexpr int argpack_mul_to_v = argpack_mul_to<index, arg1, args...>::value;

  /// @brief computes the log in base 2 of an integer.
  template<size_t x>
  struct log2 : std::integral_constant<size_t, 1+log2<x/2>::value> {};

  template<>
  struct log2<1> : std::integral_constant<size_t, 0> {};

  template<size_t x>
  inline constexpr size_t log2_v = log2<x>::value;

}  // namespace t8gpu::meta

#endif  // UTILS_META_H
