/// @file memory_manager.h
/// @brief This header file declares the MemoryManager class that
///        handles GPU memory allocation

#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <cuda/std/array>
#include <utils/meta.h>
#include <shared_device_vector.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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

    void set_volume(const thrust::device_vector<float_type>& buffer);
    void set_volume(float_type* buffer);

    float_type* get_own_volume();
    float_type const* get_own_volume() const;

    class MemoryAccessorOwn {
      cuda::std::array<float_type*, nb_variables> pointers;

    public:
      [[nodiscard]] __device__ __host__ inline float_type* get(variable_index_type i) {
	return pointers[i];
      }

      [[nodiscard]] __device__ __host__ inline float_type const* get(variable_index_type i) const {
	return pointers[i];
      }

      template<typename T1, typename T2, typename... Ts>
      [[nodiscard]] __device__ __host__ inline std::enable_if_t<t8gpu::meta::all_same_v<VariableType, T1, T2, Ts...>, std::array<float_type*, sizeof...(Ts) + 2>> get(T1 i1, T2 i2, Ts... is) {
	return { get(i1), get(i2), get(is)... };
      }

      template<typename T1, typename T2, typename... Ts>
      [[nodiscard]] __device__ __host__ inline std::enable_if_t<t8gpu::meta::all_same_v<VariableType, T1, T2, Ts...>, std::array<float_type const*, sizeof...(Ts) + 2>> get(T1 i1, T2 i2, Ts... is) const {
	return { get(i1), get(i2), get(is)... };
      }
      friend MemoryManager<VariableType, StepType>;
    };

    class MemoryAccessorAll {
      cuda::std::array<float_type* const*, nb_variables> pointers;

    public:
      [[nodiscard]] __device__ __host__ inline float_type* get(variable_index_type i) {
	return pointers[i];
      }

      [[nodiscard]] __device__ __host__ inline float_type const* get(variable_index_type i) const {
	return pointers[i];
      }

      template<typename T1, typename T2, typename... Ts>
      [[nodiscard]] __device__ __host__ inline std::enable_if_t<t8gpu::meta::all_same_v<VariableType, T1, T2, Ts...>, cuda::std::array<float_type*, sizeof...(Ts) + 2>> get(T1 i1, T2 i2, Ts... is) {
	return { get(i1), get(i2), get(is)... };
      }

      template<typename T1, typename T2, typename... Ts>
      [[nodiscard]] __device__ __host__ inline std::enable_if_t<t8gpu::meta::all_same_v<VariableType, T1, T2, Ts...>, cuda::std::array<float_type const*, sizeof...(Ts) + 2>> get(T1 i1, T2 i2, Ts... is) const {
	return { get(i1), get(i2), get(is)... };
      }
      friend MemoryManager<VariableType, StepType>;
    };

    [[nodiscard]] MemoryAccessorOwn get_own_variables(step_index_type step);
    [[nodiscard]] MemoryAccessorAll get_all_variables(step_index_type step);

  private:
    t8gpu::SharedDeviceVector<std::array<float_type, nb_variables*nb_steps+1>> m_device_buffer;
  };

} // namespace t8gpu

#include <memory_manager.inl>

#endif // MEMORY_MANAGER_H
