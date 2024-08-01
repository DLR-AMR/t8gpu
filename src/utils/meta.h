/// @file meta.h
/// @brief This header file contains metaprogramming constructs.

#ifndef UTILS_META_H
#define UTILS_META_H

#include <type_traits>

namespace t8gpu::meta {

  ///
  /// @brief all_same is a bool integral constant struct whose value
  ///        static field is true only if all of the types stripped
  ///        from cv supplied are equal. Here are a few usage
  ///        examples:
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
  struct all_same<T1, T2, Ts...> : std::integral_constant<bool, (std::is_same_v<std::remove_cv_t<T1>, std::remove_cv_t<T2>> && all_same<T2, Ts...>::value)> {};

  template<typename... Ts>
  inline constexpr bool all_same_v = all_same<Ts...>::value;


} // namespace t8gpu

#endif // UTILS_META_H
