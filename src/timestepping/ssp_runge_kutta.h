#ifndef TIMESTEPPING_SSP_RUNGE_KUTTA_H
#define TIMESTEPPING_SSP_RUNGE_KUTTA_H

namespace t8gpu::timestepping {

__global__ void SSP_3RK_step1(double const* __restrict__ rho_prev,
			      double const* __restrict__ rho_v1_prev,
			      double const* __restrict__ rho_v2_prev,
			      double const* __restrict__ rho_e_prev,
			      double* __restrict__ rho_1,
			      double* __restrict__ rho_v1_1,
			      double* __restrict__ rho_v2_1,
			      double* __restrict__ rho_e_1,
			      double const* __restrict__ volume,
			      double* __restrict__ rho_fluxes,
			      double* __restrict__ rho_v1_fluxes,
			      double* __restrict__ rho_v2_fluxes,
			      double* __restrict__ rho_e_fluxes,
			      double delta_t, int nb_elements);

__global__ void SSP_3RK_step2(double const* __restrict__ rho_prev,
			      double const* __restrict__ rho_v1_prev,
			      double const* __restrict__ rho_v2_prev,
			      double const* __restrict__ rho_e_prev,
			      double* __restrict__ rho_1,
			      double* __restrict__ rho_v1_1,
			      double* __restrict__ rho_v2_1,
			      double* __restrict__ rho_e_1,
			      double* __restrict__ rho_2,
			      double* __restrict__ rho_v1_2,
			      double* __restrict__ rho_v2_2,
			      double* __restrict__ rho_e_2,
			      double const* __restrict__ volume,
			      double* __restrict__ rho_fluxes,
			      double* __restrict__ rho_v1_fluxes,
			      double* __restrict__ rho_v2_fluxes,
			      double* __restrict__ rho_e_fluxes,
			      double delta_t, int nb_elements);

__global__ void SSP_3RK_step3(double const* __restrict__ rho_prev,
			      double const* __restrict__ rho_v1__prev,
			      double const* __restrict__ rho_v2_prev,
			      double const* __restrict__ rho_e_prev,
			      double* __restrict__ rho_2,
			      double* __restrict__ rho_v1_2,
			      double* __restrict__ rho_v2_2,
			      double* __restrict__ rho_e_2,
			      double* __restrict__ rho_next,
			      double* __restrict__ rho_v1_next,
			      double* __restrict__ rho_v2_next,
			      double* __restrict__ rho_e_next,
			      double const* __restrict__ volume,
			      double* __restrict__ rho_fluxes,
			      double* __restrict__ rho_v1_fluxes,
			      double* __restrict__ rho_v2_fluxes,
			      double* __restrict__ rho_e_fluxes,
			      double delta_t, int nb_elements);

} // namespace t8gpu

#endif // TIMESTEPPING_SSP_RUNGE_KUTTA_H
