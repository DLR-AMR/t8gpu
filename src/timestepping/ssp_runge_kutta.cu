#include <timestepping/ssp_runge_kutta.h>

__global__ void t8gpu::timestepping::SSP_3RK_step1(double const* __restrict__ u_prev, double* __restrict__ u_1,
						   double const* __restrict__ volume, double* __restrict__ fluxes, double delta_t, int nb_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_elements) return;

  u_1[i] = u_prev[i] + delta_t / volume[i] * fluxes[i];
  fluxes[i] = 0.0;
}

__global__  void t8gpu::timestepping::SSP_3RK_step2(double const* __restrict__ u_prev, double* __restrict__ u_1, double* __restrict__ u_2,
						    double const* __restrict__ volume, double* __restrict__ fluxes, double delta_t, int nb_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_elements) return;

  u_2[i] = 3.0/4.0*u_prev[i] + 1.0/4.0*u_1[i] + 1.0/4.0*delta_t / volume[i] * fluxes[i];
  fluxes[i] = 0.0;
}

__global__  void t8gpu::timestepping::SSP_3RK_step3(double const* __restrict__ u_prev, double* __restrict__ u_2, double* __restrict__ u_next,
						    double const* __restrict__ volume, double* __restrict__ fluxes, double delta_t, int nb_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_elements) return;

  u_next[i] = 1.0/3.0*u_prev[i] + 2.0/3.0*u_2[i] + 2.0/3.0*delta_t / volume[i] * fluxes[i];
  fluxes[i] = 0.0;
}
