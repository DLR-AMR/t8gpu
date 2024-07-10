#ifndef TIMESTEPPING_SSP_RUNGE_KUTTA_H
#define TIMESTEPPING_SSP_RUNGE_KUTTA_H

namespace t8gpu::timestepping {

template<typename float_type>
__global__ void SSP_3RK_step1(float_type const* __restrict__ rho_prev,
			      float_type const* __restrict__ rho_v1_prev,
			      float_type const* __restrict__ rho_v2_prev,
			      float_type const* __restrict__ rho_e_prev,
			      float_type* __restrict__ rho_1,
			      float_type* __restrict__ rho_v1_1,
			      float_type* __restrict__ rho_v2_1,
			      float_type* __restrict__ rho_e_1,
			      float_type const* __restrict__ volume,
			      float_type* __restrict__ rho_fluxes,
			      float_type* __restrict__ rho_v1_fluxes,
			      float_type* __restrict__ rho_v2_fluxes,
			      float_type* __restrict__ rho_e_fluxes,
			      float_type delta_t, int nb_elements);

template<typename float_type>
__global__ void SSP_3RK_step2(float_type const* __restrict__ rho_prev,
			      float_type const* __restrict__ rho_v1_prev,
			      float_type const* __restrict__ rho_v2_prev,
			      float_type const* __restrict__ rho_e_prev,
			      float_type* __restrict__ rho_1,
			      float_type* __restrict__ rho_v1_1,
			      float_type* __restrict__ rho_v2_1,
			      float_type* __restrict__ rho_e_1,
			      float_type* __restrict__ rho_2,
			      float_type* __restrict__ rho_v1_2,
			      float_type* __restrict__ rho_v2_2,
			      float_type* __restrict__ rho_e_2,
			      float_type const* __restrict__ volume,
			      float_type* __restrict__ rho_fluxes,
			      float_type* __restrict__ rho_v1_fluxes,
			      float_type* __restrict__ rho_v2_fluxes,
			      float_type* __restrict__ rho_e_fluxes,
			      float_type delta_t, int nb_elements);

template<typename float_type>
__global__ void SSP_3RK_step3(float_type const* __restrict__ rho_prev,
			      float_type const* __restrict__ rho_v1__prev,
			      float_type const* __restrict__ rho_v2_prev,
			      float_type const* __restrict__ rho_e_prev,
			      float_type* __restrict__ rho_2,
			      float_type* __restrict__ rho_v1_2,
			      float_type* __restrict__ rho_v2_2,
			      float_type* __restrict__ rho_e_2,
			      float_type* __restrict__ rho_next,
			      float_type* __restrict__ rho_v1_next,
			      float_type* __restrict__ rho_v2_next,
			      float_type* __restrict__ rho_e_next,
			      float_type const* __restrict__ volume,
			      float_type* __restrict__ rho_fluxes,
			      float_type* __restrict__ rho_v1_fluxes,
			      float_type* __restrict__ rho_v2_fluxes,
			      float_type* __restrict__ rho_e_fluxes,
			      float_type delta_t, int nb_elements);

} // namespace t8gpu

#include <timestepping/ssp_runge_kutta.inl>

#endif // TIMESTEPPING_SSP_RUNGE_KUTTA_H
