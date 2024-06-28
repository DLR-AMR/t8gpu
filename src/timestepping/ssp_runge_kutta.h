#ifndef TIMESTEPPING_SSP_RUNGE_KUTTA_H
#define TIMESTEPPING_SSP_RUNGE_KUTTA_H

namespace t8gpu::timestepping {

__global__ void SSP_3RK_step1(double const* __restrict__ u_prev, double* __restrict__ u_1,
			      double const* __restrict__ volume, double* __restrict__ fluxes, double delta_t, int nb_elements);

__global__ void SSP_3RK_step2(double const* __restrict__ u_prev, double* __restrict__ u_1, double* __restrict__ u_2,
			      double const* __restrict__ volume, double* __restrict__ fluxes, double delta_t, int nb_elements);

__global__ void SSP_3RK_step3(double const* __restrict__ u_prev, double* __restrict__ u_2, double* __restrict__ u_next,
			      double const* __restrict__ volume, double* __restrict__ fluxes, double delta_t, int nb_elements);

} // namespace t8gpu

#endif // TIMESTEPPING_SSP_RUNGE_KUTTA_H
