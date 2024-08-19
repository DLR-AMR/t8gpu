#ifndef TIMESTEPPING_SSP_RUNGE_KUTTA_H
#define TIMESTEPPING_SSP_RUNGE_KUTTA_H

#include <cuda/std/array>
#include <memory/memory_manager.h>

namespace t8gpu::timestepping {

  template<typename VariableType>
  __global__ void SSP_3RK_step1(MemoryAccessorOwn<VariableType> prev,
				MemoryAccessorOwn<VariableType> step1,
				MemoryAccessorOwn<VariableType> fluxes,
				typename variable_traits<VariableType>::float_type const* __restrict__ volume,
				typename variable_traits<VariableType>::float_type delta_t, int nb_elements);

  template<typename VariableType>
  __global__ void SSP_3RK_step2(MemoryAccessorOwn<VariableType> prev,
				MemoryAccessorOwn<VariableType> step1,
				MemoryAccessorOwn<VariableType> step2,
				MemoryAccessorOwn<VariableType> fluxes,
				typename variable_traits<VariableType>::float_type const* __restrict__ volume,
				typename variable_traits<VariableType>::float_type delta_t, int nb_elements);

  template<typename VariableType>
  __global__ void SSP_3RK_step3(MemoryAccessorOwn<VariableType> prev,
				MemoryAccessorOwn<VariableType> step2,
				MemoryAccessorOwn<VariableType> next,
				MemoryAccessorOwn<VariableType> fluxes,
				typename variable_traits<VariableType>::float_type const* __restrict__ volume,
				typename variable_traits<VariableType>::float_type delta_t, int nb_elements);

} // namespace t8gpu

#include <timestepping/ssp_runge_kutta.inl>

#endif // TIMESTEPPING_SSP_RUNGE_KUTTA_H
