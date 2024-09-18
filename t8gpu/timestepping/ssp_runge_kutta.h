#ifndef TIMESTEPPING_SSP_RUNGE_KUTTA_H
#define TIMESTEPPING_SSP_RUNGE_KUTTA_H

#include <t8gpu/memory/memory_manager.h>
#include <t8gpu/memory/subgrid_memory_manager.h>

namespace t8gpu::timestepping {

  /// @brief First substep of SSP-RK3 timestepping.
  ///
  /// @param [in]      prev          previous step variables.
  /// @param [out]     step1         1st substep variables.
  /// @param [in,out] fluxes        fluxes variables. The fluxes
  ///                               are also set back to 0.
  /// @param [in]      volume        local element volumes.
  /// @param [in]      delta_t       timestep.
  /// @param [in]      num_elements  number of local elements.
  template<typename VariableType>
  __global__ void SSP_3RK_step1(MemoryAccessorOwn<VariableType> prev,
                                MemoryAccessorOwn<VariableType> step1,
                                MemoryAccessorOwn<VariableType> fluxes,
                                typename variable_traits<VariableType>::float_type const* __restrict__ volume,
                                typename variable_traits<VariableType>::float_type delta_t,
                                int                                                num_elements);

  /// @brief Second substep of SSP-RK3 timestepping.
  ///
  /// @param [in]     prev          previous step variables.
  /// @param [in]     step1         1st substep variables.
  /// @param [out]    step2         2nd substep variables.
  /// @param [in,out] fluxes        fluxes variables. The fluxes
  ///                               are also set back to 0.
  /// @param [in]     volume        local element volumes.
  /// @param [in]     delta_t       timestep.
  /// @param [in]     num_elements  number of local elements.
  template<typename VariableType>
  __global__ void SSP_3RK_step2(MemoryAccessorOwn<VariableType> prev,
                                MemoryAccessorOwn<VariableType> step1,
                                MemoryAccessorOwn<VariableType> step2,
                                MemoryAccessorOwn<VariableType> fluxes,
                                typename variable_traits<VariableType>::float_type const* __restrict__ volume,
                                typename variable_traits<VariableType>::float_type delta_t,
                                int                                                num_elements);

  /// @brief Third substep of SSP-RK3 timestepping.
  ///
  /// @param [in]     prev          previous step variables.
  /// @param [in]     step2         2nd substep variables.
  /// @param [out]    next          next step variables.
  /// @param [in,out] fluxes        fluxes variables. The fluxes
  ///                               are also set back to 0.
  /// @param [in]     volume        local element volumes.
  /// @param [in]     delta_t       timestep.
  /// @param [in]     num_elements  number of local elements.
  template<typename VariableType>
  __global__ void SSP_3RK_step3(MemoryAccessorOwn<VariableType> prev,
                                MemoryAccessorOwn<VariableType> step2,
                                MemoryAccessorOwn<VariableType> next,
                                MemoryAccessorOwn<VariableType> fluxes,
                                typename variable_traits<VariableType>::float_type const* __restrict__ volume,
                                typename variable_traits<VariableType>::float_type delta_t,
                                int                                                num_elements);

  namespace subgrid {

    /// @brief First substep of SSP-RK3 timestepping.
    ///
    /// @param [in]      prev          previous step variables.
    /// @param [out]     step1         1st substep variables.
    /// @param [in,out] fluxes        fluxes variables. The fluxes
    ///                               are also set back to 0.
    /// @param [in]      volume        local element volumes.
    /// @param [in]      delta_t       timestep.
    /// @param [in]      num_elements  number of local elements.
    template<typename VariableType, typename SubgridType>
    __global__ void SSP_3RK_step1(SubgridMemoryAccessorOwn<VariableType, SubgridType> prev,
                                  SubgridMemoryAccessorOwn<VariableType, SubgridType> step1,
                                  SubgridMemoryAccessorOwn<VariableType, SubgridType> fluxes,
                                  typename variable_traits<VariableType>::float_type const* __restrict__ volumes,
                                  typename variable_traits<VariableType>::float_type delta_t);

    /// @brief Second substep of SSP-RK3 timestepping.
    ///
    /// @param [in]     prev          previous step variables.
    /// @param [in]     step1         1st substep variables.
    /// @param [out]    step2         2nd substep variables.
    /// @param [in,out] fluxes        fluxes variables. The fluxes
    ///                               are also set back to 0.
    /// @param [in]     volume        local element volumes.
    /// @param [in]     delta_t       timestep.
    /// @param [in]     num_elements  number of local elements.
    template<typename VariableType, typename SubgridType>
    __global__ void SSP_3RK_step2(SubgridMemoryAccessorOwn<VariableType, SubgridType> prev,
                                  SubgridMemoryAccessorOwn<VariableType, SubgridType> step1,
                                  SubgridMemoryAccessorOwn<VariableType, SubgridType> step2,
                                  SubgridMemoryAccessorOwn<VariableType, SubgridType> fluxes,
                                  typename variable_traits<VariableType>::float_type const* __restrict__ volumes,
                                  typename variable_traits<VariableType>::float_type delta_t);

    /// @brief Third substep of SSP-RK3 timestepping.
    ///
    /// @param [in]     prev          previous step variables.
    /// @param [in]     step2         2nd substep variables.
    /// @param [out]    next          next step variables.
    /// @param [in,out] fluxes        fluxes variables. The fluxes
    ///                               are also set back to 0.
    /// @param [in]     volume        local element volumes.
    /// @param [in]     delta_t       timestep.
    /// @param [in]     num_elements  number of local elements.
    template<typename VariableType, typename SubgridType>
    __global__ void SSP_3RK_step3(SubgridMemoryAccessorOwn<VariableType, SubgridType> prev,
                                  SubgridMemoryAccessorOwn<VariableType, SubgridType> step2,
                                  SubgridMemoryAccessorOwn<VariableType, SubgridType> next,
                                  SubgridMemoryAccessorOwn<VariableType, SubgridType> fluxes,
                                  typename variable_traits<VariableType>::float_type const* __restrict__ volumes,
                                  typename variable_traits<VariableType>::float_type delta_t);

  }  // namespace subgrid

}  // namespace t8gpu::timestepping

#include "ssp_runge_kutta.inl"

#endif  // TIMESTEPPING_SSP_RUNGE_KUTTA_H
