/// @file subgrid_memory_manager.h
/// @brief This header file declares the SubgridMemoryManager class
///        that handles GPU memory allocation as well as the
///        SubgridMemoryAccessor{Own,All} helper accessor classes for
///        the use of subgrids.

#ifndef MEMORY_SUBGRID_MEMORY_MANAGER_H
#define MEMORY_SUBGRID_MEMORY_MANAGER_H

#include <t8gpu/memory/memory_manager.h>
#include <t8gpu/memory/shared_device_vector.h>
#include <t8gpu/utils/meta.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace t8gpu {

  // Forward declarations.
  template<typename VariableType, typename SubgridType>
  class SubgridMemoryAccessorOwn;

  template<typename VariableType, typename SubgridType>
  class SubgridMemoryAccessorAll;

  /// @brief Templated type representing a subgrid.
  ///
  /// @tparam extents... The extents of the subgrid.
  ///
  /// This templated struct defines the grid dimension as well as the
  /// data access pattern (column major).
  template<int... extents>
  struct Subgrid {
    /// @brief get the rank of the subgrid (i.e. the number of dimensions).
    static constexpr int rank = sizeof...(extents);

    /// @brief the number of elements in the subgrid.
    static constexpr int size = (extents * ...);

    /// @brief get the extent of the subgrid in a given dimension.
    ///
    /// @ptaram dim The dimension for which we want to fetch the extent.
    template<int dim>
    static constexpr int extent = meta::argpack_at_v<dim, extents...>;

    /// @brief get the stride of a dimension.
    ///
    /// @tparam [in] i dimension.
    ///
    /// We use *column major* ordering to match CUDA grid ordering
    /// (i.e. the first index changes the fastest).
    template<int i>
    static constexpr int stride = meta::argpack_mul_to_v<i, extents...>;

    /// @brief from a rank-dim multi-index, retrieve the flat index.
    ///
    /// @param [in] is the indices.
    template<typename... Ts>
    static constexpr inline int flat_index(Ts... is) {
      return flat_index_impl<Ts...>(is..., std::make_integer_sequence<int, sizeof...(is)>{});
    }

    /// @brief Simple wrapper class around a array to access subgrid data.
    template<typename float_type>
    class Accessor {
     public:
      /// @brief copy constructor.
      Accessor(Accessor const& other) = default;

      /// @brief assignment operator.
      Accessor& operator=(Accessor const& other) = default;

      /// @brief Retrieve data corresponding to a subgrid element.
      ///
      /// @param [in] e_idx The local element index.
      /// @param [in] is the indices into the subgrid.
      template<typename... Ts>
      [[nodiscard]] inline __device__
          std::enable_if_t<(sizeof...(Ts) == rank) && std::conjunction_v<std::is_integral<Ts>...>, float_type&>
          operator()(size_t e_idx, Ts... is) {
        return m_data[e_idx * size + flat_index(is...)];
      }

      /// @brief Retrieve data corresponding to a subgrid element.
      ///
      /// @param [in] e_idx The local element index.
      /// @param [in] is the indices into the subgrid.
      template<typename... Ts>
      [[nodiscard]] inline __device__
          std::enable_if_t<(sizeof...(Ts) == rank) && std::conjunction_v<std::is_integral<Ts>...>, float_type const&>
          operator()(size_t e_idx, Ts... is) const {
        return m_data[e_idx * size + flat_index(is...)];
      }

      /// @brief Conversion operator to retrieve the underlying
      ///        pointer.
      explicit operator float_type*() { return m_data; }

      /// @brief Conversion operator to retrieve the underlying
      ///        pointer.
      explicit operator float_type const*() const { return m_data; }

     private:
      /// @brief Constructor accessible only to the necessary classes.
      __device__ __host__ Accessor(float_type const* data) : m_data{const_cast<float_type*>(data)} {}

      float_type* m_data;

      template<typename VariableType, typename SubgridType>
      friend class SubgridMemoryAccessorOwn;

      template<typename VariableType, typename SubgridType>
      friend class SubgridMemoryAccessorAll;

      template<typename VariableType, typename StepType, typename SubgridType>
      friend class SubgridMemoryManager;
    };

    template<typename float_type>
    using accessor_type = Accessor<float_type>;

   private:
    template<typename... Ts, int... I>
    static constexpr inline int flat_index_impl(Ts... is, std::integer_sequence<int, I...>) {
      return ((stride<I> * is) + ...);
    }
  };

  /// Forward declaration of MemoryManager class needed to make it a
  /// friend class of the SubgridMemoryAccessor{Own,All}.
  template<typename VariableType, typename StepType, typename SubgridType>
  class SubgridMemoryManager;

  /// Forward declaration of MeshManager class needed to make it a
  /// friend class of the SubgridMemoryAccessor{Own,All}.
  template<typename VariableType, typename StepType, typename SubgridType>
  class SubgridMeshManager;

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
  /// __global__ void f(SubgridMemoryAccessorOwn<Variables> m) {
  ///   auto [rho_u, rho_v] = m.get(Rho_u, Rho_v);
  ///
  ///     element index
  ///         ┌─┴─┐
  ///   rho_u(e_idx, i, j, k) = ...
  ///                └──┬──┘
  ///          subgrid coordinates
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
  template<typename VariableType, typename SubgridType>
  class SubgridMemoryAccessorOwn {
    // Friend classes that can instantiate this a class.
    template<typename VariableType_, typename StepType_, typename SubgridType_>
    friend class SubgridMemoryManager;
    template<typename VariableType_, typename StepType_, typename SubgridType_>
    friend class SubgridMeshManager;

   public:
    using variable_index_type            = typename variable_traits<VariableType>::index_type;
    using float_type                     = typename variable_traits<VariableType>::float_type;
    using subgrid_type                   = SubgridType;
    constexpr static size_t nb_variables = variable_traits<VariableType>::nb_variables;

    /// @brief copy constructor.
    SubgridMemoryAccessorOwn(SubgridMemoryAccessorOwn const& other) = default;

    /// @brief assignment operator.
    SubgridMemoryAccessorOwn& operator=(SubgridMemoryAccessorOwn const& other) = default;

    /// @brief getter function to access variable data.
    ///
    /// @param i variable index.
    ///
    /// @return device pointer to variable data.
    template<typename T>
    [[nodiscard]] __device__
        __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<T, variable_index_type>,
                                         typename SubgridType::accessor_type<float_type>>
        get(T i) {
      return typename SubgridType::accessor_type<float_type>{m_pointers[static_cast<variable_index_type>(i)]};
    }

    /// @brief getter function to access variable data.
    ///
    /// @param i variable index.
    ///
    /// @return device pointer to variable data.
    template<typename T>
    [[nodiscard]] __device__
        __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<T, variable_index_type>,
                                         typename SubgridType::accessor_type<float_type> const>
        get(T i) const {
      return typename SubgridType::accessor_type<float_type>{m_pointers[static_cast<variable_index_type>(i)]};
    }

    /// @brief getter function to access multiple variables at the
    ///        same time.
    ///
    /// @param is variable indices.
    ///
    /// @return array of variable accessor wrapper class to data.
    ///
    /// This function is meant to be used in conjunction with c++ 17
    /// structured bindings to get multiple variable at the same time.
    template<typename... Ts>
    [[nodiscard]] __device__ __host__ inline std::enable_if_t<
        t8gpu::meta::is_explicitly_convertible_to_v<typename std::tuple_element<0, std::tuple<Ts...>>::type,
                                                    variable_index_type> &&
            t8gpu::meta::all_same_v<Ts...>,
        std::array<typename SubgridType::accessor_type<float_type>, sizeof...(Ts)>>
    get(Ts... is) {
      return {get(static_cast<variable_index_type>(is))...};
    }

    /// @brief getter function to access multiple variables at the
    ///        same time.
    ///
    /// @param is variable indices.
    ///
    /// @return array of variable accessor wrapper class to data.
    ///
    /// This function is meant to be used in conjunction with c++ 17
    /// structured bindings to get multiple variable at the same time.
    template<typename... Ts>
    [[nodiscard]] __device__ __host__ inline std::enable_if_t<
        t8gpu::meta::is_explicitly_convertible_to_v<typename std::tuple_element<0, std::tuple<Ts...>>::type,
                                                    variable_index_type> &&
            t8gpu::meta::all_same_v<Ts...>,
        std::array<typename SubgridType::accessor_type<float_type> const, sizeof...(Ts)>>
    get(Ts... is) const {
      return {get(static_cast<variable_index_type>(is))...};
    }

   private:
    std::array<float_type*, nb_variables> m_pointers;

    /// @brief constructor of SubgridMemoryAccessorOwn
    ///
    /// This constructor constructs an accessor by forwarding a
    /// container type. It is intentionally by design made private so
    /// as to restrict a user from constructing such an object
    /// manually. Only the friended class part of t8gpu can construct
    /// this interface object. Moreover, this class is only accessible
    /// through member functions. This allows us to freely change
    /// implementation details.
    template<typename Container>
    SubgridMemoryAccessorOwn(Container&& array) : m_pointers(std::forward<Container>(array)) {}
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
  /// __global__ void f(SubgridMemoryAccessorOwn<Variables> m) {
  ///   auto [rho_u, rho_v] = m.get(rank, Rho_u, Rho_v);
  ///
  ///     element index
  ///         ┌─┴─┐
  ///   rho_u(e_idx, i, j, k) = ...
  ///                └──┬──┘
  ///          subgrid coordinates
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
  template<typename VariableType, typename SubgridType>
  class SubgridMemoryAccessorAll {
    // Friend classesthat can instantiate this a class.
    template<typename VT, typename ST, typename SubgridType_>
    friend class SubgridMemoryManager;
    template<typename VT, typename ST, typename SubgridType_>
    friend class SubgridMeshManager;

   public:
    using variable_index_type            = typename variable_traits<VariableType>::index_type;
    using float_type                     = typename variable_traits<VariableType>::float_type;
    constexpr static size_t nb_variables = variable_traits<VariableType>::nb_variables;

    /// @brief copy constructor.
    SubgridMemoryAccessorAll(SubgridMemoryAccessorAll const& other) = default;

    /// @brief assignment operator.
    SubgridMemoryAccessorAll& operator=(SubgridMemoryAccessorAll const& other) = default;

    /// @brief getter function to access variable data.
    ///
    /// @param rank the rank.
    /// @param i    variable index.
    ///
    /// @return an array of device pointer to variable data.
    template<typename T>
    [[nodiscard]] __device__
        __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<T, variable_index_type>,
                                         typename SubgridType::accessor_type<float_type>>
        get(int rank, T i) {
      return typename SubgridType::accessor_type<float_type>{m_pointers[static_cast<variable_index_type>(i)][rank]};
    }

    /// @brief getter function to access variable data.
    ///
    /// @param rank the rank.
    /// @param i    variable index.
    ///
    /// @return an array of device pointer to variable data.
    template<typename T>
    [[nodiscard]] __device__
        __host__ inline std::enable_if_t<t8gpu::meta::is_explicitly_convertible_to_v<T, variable_index_type>,
                                         typename SubgridType::accessor_type<float_type> const>
        get(int rank, T i) const {
      return typename SubgridType::accessor_type<float_type>{m_pointers[static_cast<variable_index_type>(i)][rank]};
    }

    /// @brief getter function to access multiple variables at the
    ///        same time.
    ///
    /// @param rank  the rank.
    /// @param is    variable indices.
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
        std::array<typename SubgridType::accessor_type<float_type>, sizeof...(Ts)>>
    get(int rank, Ts... is) {
      return {get(rank, static_cast<variable_index_type>(is))...};
    }

    /// @brief getter function to access multiple variables at the
    ///        same time.
    ///
    /// @param rank  the rank.
    /// @param is    variable indices.
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
        std::array<typename SubgridType::accessor_type<float_type> const, sizeof...(Ts)>>
    get(int rank, Ts... is) const {
      return {get(rank, static_cast<variable_index_type>(is))...};
    }

   private:
    std::array<float_type* const*, nb_variables> m_pointers;

    /// @brief constructor of SubgridMemoryAccessorAll
    ///
    /// This constructor constructs an accessor by forwarding a
    /// container type. It is intentionally by design made private so
    /// as to restrict a user from constructing such an object
    /// manually. Only the friended class part of t8gpu can construct
    /// this interface object. Moreover, this class is only accessible
    /// through member functions. This allows us to freely change
    /// implementation details.
    template<typename Container>
    SubgridMemoryAccessorAll(Container&& array) : m_pointers(std::forward<Container>(array)) {}
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
  template<typename VariableType, typename StepType, typename SubgridType>
  class SubgridMemoryManager {
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
    SubgridMemoryManager(size_t nb_elements = 0, sc_MPI_Comm comm = sc_MPI_COMM_WORLD);

    ~SubgridMemoryManager() = default;

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
    [[nodiscard]] SubgridMemoryAccessorOwn<VariableType, SubgridType> get_own_variables(step_index_type step);

    /// @brief get all variables owned by all ranks.
    ///
    /// @param [in]   step the step from which we retrieve the variables.
    ///
    /// @return An object from which we can query the variables on the
    ///         CPU/GPU (but only access the data on the GPU).
    [[nodiscard]] SubgridMemoryAccessorAll<VariableType, SubgridType> get_all_variables(step_index_type step);

    /// @brief get variable owned by this rank.
    ///
    /// @param [in]   step     the step from which we retrieve the variables.
    /// @param [in]   variable the name of the variable.
    ///
    /// @return A pointer to GPU memory containing the variable data.
    [[nodiscard]] typename SubgridType::accessor_type<float_type> get_own_variable(step_index_type     step,
                                                                                   variable_index_type variable);

    /// @brief get variable owned by this rank.
    ///
    /// @param [in]   step     the step from which we retrieve the variables.
    /// @param [in]   variable the name of the variable.
    ///
    /// @return A pointer to GPU memory containing the variable data.
    [[nodiscard]] typename SubgridType::accessor_type<float_type> const get_own_variable(
        step_index_type step, variable_index_type variable) const;

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
    t8gpu::SharedDeviceVector<std::array<float_type, nb_variables * nb_steps>> m_device_buffer;
    t8gpu::SharedDeviceVector<float_type>                                      m_device_volume;
  };

}  // namespace t8gpu

#include "subgrid_memory_manager.inl"

#endif  // MEMORY_SUBGRID_MEMORY_MANAGER_H
