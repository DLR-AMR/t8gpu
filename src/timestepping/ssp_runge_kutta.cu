#include <timestepping/ssp_runge_kutta.h>

__global__ void t8gpu::timestepping::SSP_3RK_step1(double const* __restrict__ rho_prev,
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
						   double delta_t, int nb_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_elements) return;

  rho_1[i]    = rho_prev[i]    + delta_t / volume[i] * rho_fluxes[i];
  rho_v1_1[i] = rho_v1_prev[i] + delta_t / volume[i] * rho_v1_fluxes[i];
  rho_v2_1[i] = rho_v2_prev[i] + delta_t / volume[i] * rho_v2_fluxes[i];
  rho_e_1[i]  = rho_e_prev[i]  + delta_t / volume[i] * rho_e_fluxes[i];

  rho_fluxes[i]    = 0.0;
  rho_v1_fluxes[i] = 0.0;
  rho_v2_fluxes[i] = 0.0;
  rho_e_fluxes[i]  = 0.0;
}

__global__  void t8gpu::timestepping::SSP_3RK_step2(double const* __restrict__ rho_prev,
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
						    double delta_t, int nb_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_elements) return;

  rho_2[i]    = 3.0/4.0*rho_prev[i]    + 1.0/4.0*rho_1[i]    + 1.0/4.0*delta_t / volume[i] * rho_fluxes[i];
  rho_v1_2[i] = 3.0/4.0*rho_v1_prev[i] + 1.0/4.0*rho_v1_1[i] + 1.0/4.0*delta_t / volume[i] * rho_v1_fluxes[i];
  rho_v2_2[i] = 3.0/4.0*rho_v2_prev[i] + 1.0/4.0*rho_v2_1[i] + 1.0/4.0*delta_t / volume[i] * rho_v2_fluxes[i];
  rho_e_2[i]  = 3.0/4.0*rho_e_prev[i]  + 1.0/4.0*rho_e_1[i]  + 1.0/4.0*delta_t / volume[i] * rho_e_fluxes[i];

  rho_fluxes[i]    = 0.0;
  rho_v1_fluxes[i] = 0.0;
  rho_v2_fluxes[i] = 0.0;
  rho_e_fluxes[i]  = 0.0;
}

__global__  void t8gpu::timestepping::SSP_3RK_step3(double const* __restrict__ rho_prev,
						    double const* __restrict__ rho_v1_prev,
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
						    double delta_t, int nb_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_elements) return;

  rho_next[i]    = 1.0/3.0*rho_prev[i]    + 2.0/3.0*rho_2[i]    + 2.0/3.0*delta_t / volume[i] * rho_fluxes[i];
  rho_v1_next[i] = 1.0/3.0*rho_v1_prev[i] + 2.0/3.0*rho_v1_2[i] + 2.0/3.0*delta_t / volume[i] * rho_v1_fluxes[i];
  rho_v2_next[i] = 1.0/3.0*rho_v2_prev[i] + 2.0/3.0*rho_v2_2[i] + 2.0/3.0*delta_t / volume[i] * rho_v2_fluxes[i];
  rho_e_next[i]  = 1.0/3.0*rho_e_prev[i]  + 2.0/3.0*rho_e_2[i]  + 2.0/3.0*delta_t / volume[i] * rho_e_fluxes[i];

  rho_fluxes[i]    = 0.0;
  rho_v1_fluxes[i] = 0.0;
  rho_v2_fluxes[i] = 0.0;
  rho_e_fluxes[i]  = 0.0;
}
