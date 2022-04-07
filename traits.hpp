/*!
 * \file traits.hpp
 * \author Han Luo (luohancfd AT github)
 * \brief Wrapper of MPI functions
 * \version 0.1
 * \date 2020-12-22
 * \copyright GPL-3.0
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once
#ifndef TRAITS_HPP
#define TRAITS_HPP

#include <sstream>
#include <string>
#include <type_traits>

template <class T>
struct dependent_false : std::false_type {};

template <class T>
struct dependent_true : std::true_type {};

template <typename T>
struct is_string : std::false_type {};

template <>
struct is_string<std::string> : std::true_type {};

template <typename>
struct is_pair : std::false_type {};

template <typename T, typename U>
struct is_pair<std::pair<T, U>> : std::true_type {};

template <typename>
struct is_tuple : std::false_type {};
template <typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};

template <typename T>
struct is_numeric
    : std::integral_constant<bool, std::is_integral<T>::value ||
                                       std::is_floating_point<T>::value> {};

template <bool, class T = void>
struct enable_if_c {
  typedef T type;
};

template <class T>
struct enable_if_c<false, T> {};

template <class Cond, class T = void>
struct enable_if : public enable_if_c<Cond::value, T> {};

template <bool, class T = void>
struct disable_if_c {
  typedef T type;
};

template <class T>
struct disable_if_c<true, T> {};

template <class Cond, class T = void>
struct disable_if : public disable_if_c<Cond::value, T> {};

template <typename S, typename T>
struct is_streamable {
  template <typename SS, typename TT>
  static auto test(int)
      -> decltype(std::declval<SS &>() << std::declval<TT>(), std::true_type());

  template <typename, typename>
  static auto test(...) -> std::false_type;

  static const bool value = decltype(test<S, T>(0))::value;
};

template <typename Key, bool Streamable>
struct streamable_to_string {
  static std::string impl(const Key &key) {
    std::stringstream ss;
    ss << key;
    return ss.str();
  }
};

template <typename Key>
struct streamable_to_string<Key, false> {
  static std::string impl(const Key &) { return ""; }
};
#endif
