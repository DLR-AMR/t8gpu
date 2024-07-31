#ifndef TIMESTEPPING_SSP_RUNGE_KUTTA_H
#define TIMESTEPPING_SSP_RUNGE_KUTTA_H

#include <cuda/std/array>

namespace t8gpu::timestepping {

  template<typename float_type, size_t nb_variables>
  __global__ void SSP_3RK_step1(cuda::std::array<float_type* __restrict__, nb_variables> prev,
				cuda::std::array<float_type* __restrict__, nb_variables> step1,
				float_type const* __restrict__ volume,
				cuda::std::array<float_type* __restrict__, nb_variables> fluxes,
				float_type delta_t, int nb_elements);

  template<typename float_type, size_t nb_variables>
__global__ void SSP_3RK_step2(cuda::std::array<float_type* __restrict__, nb_variables> prev,
			      cuda::std::array<float_type* __restrict__, nb_variables> step1,
			      cuda::std::array<float_type* __restrict__, nb_variables> step2,
			      float_type const* __restrict__ volume,
			      cuda::std::array<float_type* __restrict__, nb_variables> fluxes,
			      float_type delta_t, int nb_elements);

  template<typename float_type, size_t nb_variables>
__global__ void SSP_3RK_step3(cuda::std::array<float_type* __restrict__, nb_variables> prev,
			      cuda::std::array<float_type* __restrict__, nb_variables> step2,
			      cuda::std::array<float_type* __restrict__, nb_variables> next,
			      float_type const* __restrict__ volume,
			      cuda::std::array<float_type* __restrict__, nb_variables> fluxes,
			      float_type delta_t, int nb_elements);

} // namespace t8gpu

#include <timestepping/ssp_runge_kutta.inl>

#endif // TIMESTEPPING_SSP_RUNGE_KUTTA_H
