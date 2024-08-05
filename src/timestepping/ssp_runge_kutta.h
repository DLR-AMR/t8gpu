#ifndef TIMESTEPPING_SSP_RUNGE_KUTTA_H
#define TIMESTEPPING_SSP_RUNGE_KUTTA_H

#include <cuda/std/array>
#include <memory_manager.h>

namespace t8gpu::timestepping {

  template<typename VariableType, typename StepType>
  __global__ void SSP_3RK_step1(typename t8gpu::MemoryManager<VariableType, StepType>::MemoryAccessorOwn prev,
				typename t8gpu::MemoryManager<VariableType, StepType>::MemoryAccessorOwn step1,
				typename t8gpu::MemoryManager<VariableType, StepType>::MemoryAccessorOwn fluxes,
				typename t8gpu::MemoryManager<VariableType, StepType>::float_type const* __restrict__ volume,
				typename t8gpu::MemoryManager<VariableType, StepType>::float_type delta_t, int nb_elements);

  template<typename VariableType, typename StepType>
  __global__ void SSP_3RK_step2(typename t8gpu::MemoryManager<VariableType, StepType>::MemoryAccessorOwn prev,
				typename t8gpu::MemoryManager<VariableType, StepType>::MemoryAccessorOwn step1,
				typename t8gpu::MemoryManager<VariableType, StepType>::MemoryAccessorOwn step2,
				typename t8gpu::MemoryManager<VariableType, StepType>::MemoryAccessorOwn fluxes,
				typename t8gpu::MemoryManager<VariableType, StepType>::float_type const* __restrict__ volume,
				typename t8gpu::MemoryManager<VariableType, StepType>::float_type delta_t, int nb_elements);

  template<typename VariableType, typename StepType>
  __global__ void SSP_3RK_step3(typename t8gpu::MemoryManager<VariableType, StepType>::MemoryAccessorOwn prev,
				typename t8gpu::MemoryManager<VariableType, StepType>::MemoryAccessorOwn step2,
				typename t8gpu::MemoryManager<VariableType, StepType>::MemoryAccessorOwn next,
				typename t8gpu::MemoryManager<VariableType, StepType>::MemoryAccessorOwn fluxes,
				typename t8gpu::MemoryManager<VariableType, StepType>::float_type const* __restrict__ volume,
				typename t8gpu::MemoryManager<VariableType, StepType>::float_type delta_t, int nb_elements);

} // namespace t8gpu

#include <timestepping/ssp_runge_kutta.inl>

#endif // TIMESTEPPING_SSP_RUNGE_KUTTA_H
