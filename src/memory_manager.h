/// @file memory_manager.h
/// @brief This header file declares the MemoryManager class that
///        handles GPU memory allocation

#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <utils/meta.h>
#include <shared_device_vector.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <tuple>
#include <type_traits>

namespace t8gpu {

  // Type traits are necessary to be able to choose a VariableList type
  // that is not a struct and thus cannot have static member and/or
  // type-aliases (e.g. enums). For now, we are using enum for the
  // variable list type, but we may change that later on to have
  // something more flexible. This variables_traits can allow to still
  // retain old behavior while allowing new types of VariableLists.
  template<class VariableList, typename = void>
  struct variable_traits {};

  template<class VariableType>
  struct variable_traits<VariableType, typename std::enable_if_t<std::is_enum_v<VariableType>>> {
    using float_type = float;
    using index_type = VariableType;
    static constexpr size_t nb_variables = VariableType::nb_variables;
  };

  template<class StepList, typename = void>
  struct step_traits {};

  template<class StepType>
  struct step_traits<StepType, typename std::enable_if_t<std::is_enum_v<StepType>>> {
    using float_type = float;
    using index_type = StepType;
    static constexpr size_t nb_steps = StepType::nb_steps;
  };

  ///
  /// @brief A class that manages GPU memory templated on the variable enum
  ///        class listing and step.
  ///
  template<typename VariableType, typename StepType>
  class MemoryManager {
  public:
    using float_type = typename variable_traits<VariableType>::float_type;
    using variable_index_type = typename variable_traits<VariableType>::index_type;
    static constexpr size_t nb_variables = variable_traits<VariableType>::nb_variables;

    using step_index_type = typename step_traits<StepType>::index_type;
    static constexpr size_t nb_steps = step_traits<StepType>::nb_steps;

    MemoryManager(size_t nb_elements = 0);
    ~MemoryManager() = default;

    inline void resize(size_t new_size);

    void set_variable(step_index_type step, variable_index_type variable, const thrust::device_vector<float_type>& buffer);
    void set_variable(step_index_type step, variable_index_type variable, const thrust::host_vector<float_type>& buffer);
    void set_variable(step_index_type step, variable_index_type variable, float_type* buffer);

    void set_volume(const thrust::host_vector<float_type>& buffer);
    void set_volume(const thrust::device_vector<float_type>& buffer);
    void set_volume(float_type* buffer);

    float_type* get_own_volume();
    float_type const* get_own_volume() const;

    float_type* const* get_all_volume();
    float_type const* const* get_all_volume() const;

    class MemoryAccessorOwn {
      std::array<float_type*, nb_variables> pointers;

    public:
      template<typename T>
      [[nodiscard]] __device__ __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<T, variable_index_type>, float_type*> get(T i) {
	return pointers[static_cast<variable_index_type>(i)];
      }

      template<typename T>
      [[nodiscard]] __device__ __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<T, variable_index_type>, float_type const*> get(T i) const {
	return pointers[static_cast<variable_index_type>(i)];
      }

      template<typename... Ts>
      [[nodiscard]] __device__ __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<typename std::tuple_element<0, std::tuple<Ts...>>::type, variable_index_type> && t8gpu::meta::all_same_v<Ts...>, std::array<float_type*, sizeof...(Ts)>> get(Ts... is) {
	return { get(static_cast<variable_index_type>(is))... };
      }

      template<typename... Ts>
      [[nodiscard]] __device__ __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<typename std::tuple_element<0, std::tuple<Ts...>>::type, variable_index_type> && t8gpu::meta::all_same_v<Ts...>, std::array<float_type const*, sizeof...(Ts)>> get(Ts... is) const {
	return {  get(static_cast<variable_index_type>(is))... };
      }
      friend MemoryManager<VariableType, StepType>;
    };

    class MemoryAccessorAll {
      std::array<float_type* const*, nb_variables> pointers;

    public:
      template<typename T>
      [[nodiscard]] __device__ __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<T, variable_index_type>, float_type* const*> get(T i) {
	return pointers[static_cast<variable_index_type>(i)];
      }

      template<typename T>
      [[nodiscard]] __device__ __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<T, variable_index_type>, float_type const* const*> get(T i) const {
	return pointers[static_cast<variable_index_type>(i)];
      }

      template<typename... Ts>
      [[nodiscard]] __device__ __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<typename std::tuple_element<0, std::tuple<Ts...>>::type, variable_index_type> && t8gpu::meta::all_same_v<Ts...>, std::array<float_type* const*, sizeof...(Ts)>> get(Ts... is) {
	return { get(static_cast<variable_index_type>(is))... };
      }

      template<typename... Ts>
      [[nodiscard]] __device__ __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<typename std::tuple_element<0, std::tuple<Ts...>>::type, variable_index_type> && t8gpu::meta::all_same_v<Ts...>, std::array<float_type const* const*, sizeof...(Ts)>> get(Ts... is) const {
	return { get(static_cast<variable_index_type>(is))... };
      }

      friend MemoryManager<VariableType, StepType>;
    };

    [[nodiscard]] MemoryAccessorOwn get_own_variables(step_index_type step);
    [[nodiscard]] MemoryAccessorAll get_all_variables(step_index_type step);

    [[nodiscard]] float_type* get_own_variable(step_index_type step, variable_index_type variable);
    [[nodiscard]] float_type const* get_own_variable(step_index_type step, variable_index_type variable) const;

  private:
    t8gpu::SharedDeviceVector<std::array<float_type, nb_variables*nb_steps+1>> m_device_buffer;
  };

} // namespace t8gpu

#include <memory_manager.inl>

#endif // MEMORY_MANAGER_H
