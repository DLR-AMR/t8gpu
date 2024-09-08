/// @file memory_manager.h
/// @brief This header file declares the MemoryManager class that
///        handles GPU memory allocation as well as the
///        MemoryAccessor{Own,All} helper accessor classes.

#ifndef MEMORY_MEMORY_MANAGER_H
#define MEMORY_MEMORY_MANAGER_H

#include <t8gpu/memory/shared_device_vector.h>
#include <t8gpu/utils/meta.h>
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
    using float_type                     = double;
    using index_type                     = VariableType;
    static constexpr size_t nb_variables = VariableType::nb_variables;
  };

  template<class StepList, typename = void>
  struct step_traits {};

  template<class StepType>
  struct step_traits<StepType, typename std::enable_if_t<std::is_enum_v<StepType>>> {
    using float_type                 = float;
    using index_type                 = StepType;
    static constexpr size_t nb_steps = StepType::nb_steps;
  };

  /// Forward declaration of MemoryManager class needed to make it a
  /// friend class of the MemoryAccessor{Own,All}.
  template<typename VariableType, typename StepType>
  class MemoryManager;

  /// Forward declaration of MemoryManager class needed to make it a
  /// friend class of the MemoryAccessor{Own,All}.
  template<typename VariableType, typename StepType, typename SubgridType>
  class SubgridMemoryManager;

  /// Forward declaration of MeshManager class needed to make it a
  /// friend class of the MemoryAccessor{Own,All}.
  template<typename VariableType, typename StepType, size_t dim>
  class MeshManager;

  ///
  /// @brief A class that is used to access variables on the CPU and
  ///        GPU side using their name or index.
  ///
  /// This struct gives access to the variable data owned by the rank
  /// using the get member functions. Example using the following
  /// variable list type:
  ///
  /// enum Variables {Rho, Rho_u, Rho_v, Rho_e, nb_variables };
  ///
  /// Either select the variables with their names:
  ///
  /// __global__ void f(MemoryAccessorOwn<Variables> m) {
  ///   auto [rho_u, rho_v] = m.get(Rho_u, Rho_v);
  ///
  ///   rho_u[i] = ...
  ///
  /// ...
  /// }
  ///
  /// Or, iterate on all of the variables:
  ///
  /// __global__ void f(MemoryAccessorOwn<Variables> m) {
  ///
  ///   for (int k=0; k<nb_variables; k++) {
  ///     m.get(k)[i] = ...
  ///   }
  /// }
  template<typename VariableType>
  class MemoryAccessorOwn {
    // Friend classesthat can instantiate this a class.
    template<typename VT, typename ST>
    friend class MemoryManager;
    template<typename VT, typename ST, size_t dim_>
    friend class MeshManager;
    template<typename VT, typename ST, typename SubgridType>
    friend class SubgridMeshManager;

   public:
    using variable_index_type            = typename variable_traits<VariableType>::index_type;
    using float_type                     = typename variable_traits<VariableType>::float_type;
    constexpr static size_t nb_variables = variable_traits<VariableType>::nb_variables;

    /// @brief copy constructor.
    MemoryAccessorOwn(MemoryAccessorOwn const& other) = default;

    /// @brief assignment operator.
    MemoryAccessorOwn& operator=(MemoryAccessorOwn const& other) = default;

    /// @brief getter function to access variable data.
    ///
    /// @param i variable index.
    ///
    /// @return device pointer to variable data.
    template<typename T>
    [[nodiscard]] __device__
        __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<T, variable_index_type>,
                                         float_type*>
        get(T i) {
      return m_pointers[static_cast<variable_index_type>(i)];
    }

    /// @brief getter function to access variable data.
    ///
    /// @param i variable index.
    ///
    /// @return device pointer to variable data.
    template<typename T>
    [[nodiscard]] __device__
        __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<T, variable_index_type>,
                                         float_type const*>
        get(T i) const {
      return m_pointers[static_cast<variable_index_type>(i)];
    }

    /// @brief getter function to access multiple variables at the
    ///        same time.
    ///
    /// @param is variable indices.
    ///
    /// @return array of device pointers to variable data.
    ///
    /// This function is meant to be used in conjunction with c++ 17
    /// structured bindings to get multiple variable at the same time.
    template<typename... Ts>
    [[nodiscard]] __device__ __host__ inline std::enable_if_t<
        t8gpu::meta::is_explicitly_convertible_to_v<typename std::tuple_element<0, std::tuple<Ts...>>::type,
                                                    variable_index_type> &&
            t8gpu::meta::all_same_v<Ts...>,
        std::array<float_type*, sizeof...(Ts)>>
    get(Ts... is) {
      return {get(static_cast<variable_index_type>(is))...};
    }

    /// @brief getter function to access multiple variables at the
    ///        same time.
    ///
    /// @param is variable indices.
    ///
    /// @return array of device pointers to variable data.
    ///
    /// This function is meant to be used in conjunction with c++ 17
    /// structured bindings to get multiple variable at the same time.
    template<typename... Ts>
    [[nodiscard]] __device__ __host__ inline std::enable_if_t<
        t8gpu::meta::is_explicitly_convertible_to_v<typename std::tuple_element<0, std::tuple<Ts...>>::type,
                                                    variable_index_type> &&
            t8gpu::meta::all_same_v<Ts...>,
        std::array<float_type const*, sizeof...(Ts)>>
    get(Ts... is) const {
      return {get(static_cast<variable_index_type>(is))...};
    }

   private:
    std::array<float_type*, nb_variables> m_pointers;

    /// @brief constructor of MemoryAccessorOwn
    ///
    /// This constructor constructs an accessor by forwarding a
    /// container type. It is intentionally by design made private so
    /// as to restrict a user from constructing such an object
    /// manually. Only the friended class part of t8gpu can construct
    /// this interface object. Moreover, this class is only accessible
    /// through member functions. This allows us to freely change
    /// implementation details.
    template<typename Container>
    MemoryAccessorOwn(Container&& array) : m_pointers(std::forward<Container>(array)) {}
  };

  ///
  /// @brief A class that is used to access variables on the CPU and
  ///        GPU side using their name or index.
  ///
  /// This struct gives access to the variable data owned by all ranks
  /// using the get member functions. Example using the following
  /// variable list type:
  ///
  /// enum Variables {Rho, Rho_u, Rho_v, Rho_e, nb_variables };
  ///
  /// Either select the variables with their names:
  ///
  /// __global__ void f(MemoryAccessorOwn<Variables> m) {
  ///   auto [rho_u, rho_v] = m.get(Rho_u, Rho_v);
  ///
  ///   rho_u[i][rank] = ...
  ///
  /// ...
  /// }
  ///
  /// Or, iterate on all of the variables:
  ///
  /// __global__ void f(MemoryAccessorOwn<Variables> m) {
  ///
  ///   for (int k=0; k<nb_variables; k++) {
  ///     m.get(k)[rank][i] = ...
  ///   }
  /// }
  template<typename VariableType>
  class MemoryAccessorAll {
    // Friend classesthat can instantiate this a class.
    template<typename VT, typename ST>
    friend class MemoryManager;
    template<typename VT, typename ST, size_t dim_>
    friend class MeshManager;

   public:
    using variable_index_type            = typename variable_traits<VariableType>::index_type;
    using float_type                     = typename variable_traits<VariableType>::float_type;
    constexpr static size_t nb_variables = variable_traits<VariableType>::nb_variables;

    /// @brief copy constructor.
    MemoryAccessorAll(MemoryAccessorAll const& other) = default;

    /// @brief assignment operator.
    MemoryAccessorAll& operator=(MemoryAccessorAll const& other) = default;

    /// @brief getter function to access variable data.
    ///
    /// @param i variable index.
    ///
    /// @return an array of device pointer to variable data.
    template<typename T>
    [[nodiscard]] __device__
        __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<T, variable_index_type>,
                                         float_type* const*>
        get(T i) {
      return m_pointers[static_cast<variable_index_type>(i)];
    }

    /// @brief getter function to access variable data.
    ///
    /// @param i variable index.
    ///
    /// @return an array of device pointer to variable data.
    template<typename T>
    [[nodiscard]] __device__
        __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<T, variable_index_type>,
                                         float_type const* const*>
        get(T i) const {
      return m_pointers[static_cast<variable_index_type>(i)];
    }

    /// @brief getter function to access multiple variables at the
    ///        same time.
    ///
    /// @param is variable indices.
    ///
    /// @return array of arrays to device pointers to variable data.
    ///
    /// This function is meant to be used in conjunction with c++ 17
    /// structured bindings to get multiple variable at the same time.
    template<typename... Ts>
    [[nodiscard]] __device__ __host__ inline std::enable_if_t<
        t8gpu::meta::is_explicitly_convertible_to_v<typename std::tuple_element<0, std::tuple<Ts...>>::type,
                                                    variable_index_type> &&
            t8gpu::meta::all_same_v<Ts...>,
        std::array<float_type* const*, sizeof...(Ts)>>
    get(Ts... is) {
      return {get(static_cast<variable_index_type>(is))...};
    }

    /// @brief getter function to access multiple variables at the
    ///        same time.
    ///
    /// @param is variable indices.
    ///
    /// @return array of arrays to device pointers to variable data.
    ///
    /// This function is meant to be used in conjunction with c++ 17
    /// structured bindings to get multiple variable at the same time.
    template<typename... Ts>
    [[nodiscard]] __device__ __host__ inline std::enable_if_t<
        t8gpu::meta::is_explicitly_convertible_to_v<typename std::tuple_element<0, std::tuple<Ts...>>::type,
                                                    variable_index_type> &&
            t8gpu::meta::all_same_v<Ts...>,
        std::array<float_type const* const*, sizeof...(Ts)>>
    get(Ts... is) const {
      return {get(static_cast<variable_index_type>(is))...};
    }

   private:
    std::array<float_type* const*, nb_variables> m_pointers;

    /// @brief constructor of MemoryAccessorAll
    ///
    /// This constructor constructs an accessor by forwarding a
    /// container type. It is intentionally by design made private so
    /// as to restrict a user from constructing such an object
    /// manually. Only the friended class part of t8gpu can construct
    /// this interface object. Moreover, this class is only accessible
    /// through member functions. This allows us to freely change
    /// implementation details.
    template<typename Container>
    MemoryAccessorAll(Container&& array) : m_pointers(std::forward<Container>(array)) {}
  };

  ///
  /// @brief A class that manages GPU memory templated on the variable enum
  ///        class listing and step.
  ///
  /// Giving a enum VariableType and StepType, this class handles GPU
  /// memory allocation for all the variables and steps. All of the
  /// variables can be accesses on the GPU from all ranks (even
  /// accessing variable data from element that the rank does not own)
  /// using the appropriate get_{own,all}_variables{s} getter
  /// functions and the variable names given by the VariableType enum
  /// and StepType.
  template<typename VariableType, typename StepType>
  class MemoryManager {
   public:
    using float_type                     = typename variable_traits<VariableType>::float_type;
    using variable_index_type            = typename variable_traits<VariableType>::index_type;
    static constexpr size_t nb_variables = variable_traits<VariableType>::nb_variables;

    using step_index_type            = typename step_traits<StepType>::index_type;
    static constexpr size_t nb_steps = step_traits<StepType>::nb_steps;

    /// @brief Constructor of the MemoryManager class.
    ///
    /// @param [in]   nb_elements specifies the initial number of elements.
    /// @param [in]   comm        specifies the MPI communicator to use.
    ///
    MemoryManager(size_t nb_elements = 0, sc_MPI_Comm comm = sc_MPI_COMM_WORLD);

    ~MemoryManager() = default;

    /// @brief set a variable.
    ///
    /// @param [in]   step     the step for which the variable is set.
    /// @param [in]   variable the name of the variable to set.
    /// @param [in]   buffer   the GPU buffer to copy from.
    void set_variable(step_index_type                          step,
                      variable_index_type                      variable,
                      thrust::device_vector<float_type> const& buffer);

    /// @brief set a variable.
    ///
    /// @param [in]   step     the step for which the variable is set.
    /// @param [in]   variable the name of the variable to set.
    /// @param [in]   buffer   the CPU buffer to copy from.
    void set_variable(step_index_type                        step,
                      variable_index_type                    variable,
                      thrust::host_vector<float_type> const& buffer);

    /// @brief set a variable.
    ///
    /// @param [in]   step     the step for which the variable is set.
    /// @param [in]   variable the name of the variable to set.
    /// @param [in]   buffer   the CPU buffer to copy from (raw pointer).
    void set_variable(step_index_type step, variable_index_type variable, float_type* buffer);

    /// @brief set the volume variable.
    ///
    /// @param [in]   buffer   the CPU buffer to copy from.
    void set_volume(thrust::host_vector<float_type> const& buffer);

    /// @brief set the volume variable.
    ///
    /// @param [in]   buffer   the GPU buffer to copy from.
    void set_volume(thrust::device_vector<float_type> const& buffer);

    /// @brief set the volume variable.
    ///
    /// @param [in]   buffer   the GPU buffer to copy from (raw pointer).
    ///
    /// @warning the buffer needs to be of size the number of elements.
    void set_volume(float_type* buffer);

    /// @brief get the volume variable of elements owned by this rank.
    ///
    /// @return A pointer to GPU memory containing the volume data.
    float_type* get_own_volume();

    /// @brief get the volume variable of elements owned by this rank.
    ///
    /// @return A pointer to GPU memory containing the volume data.
    float_type const* get_own_volume() const;

    /// @brief get the volume variable of elements owned by all ranks.
    ///
    /// @return An array of pointers to GPU memory containing the
    ///         volume data for each ranks.
    float_type* const* get_all_volume();

    /// @brief get the volume variable of elements owned by all ranks.
    ///
    /// @return An array of pointers to GPU memory containing the
    ///         volume data for each ranks.
    float_type const* const* get_all_volume() const;

    /// @brief get all variables owned by this rank.
    ///
    /// @param [in]   step the step from which we retrieve the variables.
    ///
    /// @return An object from which we can query the variables on the
    ///         CPU/GPU (but only access the data on the GPU).
    [[nodiscard]] MemoryAccessorOwn<VariableType> get_own_variables(step_index_type step);

    /// @brief get all variables owned by all ranks.
    ///
    /// @param [in]   step the step from which we retrieve the variables.
    ///
    /// @return An object from which we can query the variables on the
    ///         CPU/GPU (but only access the data on the GPU).
    [[nodiscard]] MemoryAccessorAll<VariableType> get_all_variables(step_index_type step);

    /// @brief get variable owned by this rank.
    ///
    /// @param [in]   step     the step from which we retrieve the variables.
    /// @param [in]   variable the name of the variable.
    ///
    /// @return A pointer to GPU memory containing the variable data.
    [[nodiscard]] float_type* get_own_variable(step_index_type step, variable_index_type variable);

    /// @brief get variable owned by this rank.
    ///
    /// @param [in]   step     the step from which we retrieve the variables.
    /// @param [in]   variable the name of the variable.
    ///
    /// @return A pointer to GPU memory containing the variable data.
    [[nodiscard]] float_type const* get_own_variable(step_index_type step, variable_index_type variable) const;

    /// @brief resize the variables.
    ///
    /// @param [in]   new_size new size to use.
    ///
    /// It is important to state that this function needs to be called
    /// by all MPI ranks. Otherwise this results in a lock. If a ranks
    /// does not need to change the size of the allocation, use resize
    /// with the current size of the vector. This functions may not
    /// need to reallocate even if new_size > size the copy over as
    /// the capacity of the allocation might be greater than new_size
    /// and if not we overallocate by a factor 1/2 to minimize
    /// reallocation on later calls to the resize function. It is
    /// important to note that all previously variable data is
    /// discarded upon a resize. To properly set new values, an
    /// explicit copy beforehand is needed to restore properly
    /// variable data (and do for instance interpolation).
    inline void resize(size_t new_size);

   private:
    t8gpu::SharedDeviceVector<std::array<float_type, nb_variables * nb_steps + 1>> m_device_buffer;
  };

}  // namespace t8gpu

#include "memory_manager.inl"

#endif  // MEMORY_MEMORY_MANAGER_H
