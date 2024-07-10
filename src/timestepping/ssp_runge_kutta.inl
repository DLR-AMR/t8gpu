#include <timestepping/ssp_runge_kutta.h>

template<typename ft>
struct rk_coeffs {};

template<>
struct rk_coeffs<float> {
  static constexpr float stage_2_1 = 0.75f;
  static constexpr float stage_2_2 = 0.25f;
  static constexpr float stage_2_3 = 0.25f;

  static constexpr float stage_3_1 = 0.33333333333333f;
  static constexpr float stage_3_2 = 0.66666666666666f;
  static constexpr float stage_3_3 = 0.66666666666666f;
};

template<>
struct rk_coeffs<double> {
  static constexpr double stage_2_1 = 0.75;
  static constexpr double stage_2_2 = 0.25;
  static constexpr double stage_2_3 = 0.25;

  static constexpr double stage_3_1 = 0.33333333333333;
  static constexpr double stage_3_2 = 0.66666666666666;
  static constexpr double stage_3_3 = 0.66666666666666;
};

template<typename float_type>
__global__ void t8gpu::timestepping::SSP_3RK_step1(float_type const* __restrict__ rho_prev,
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
						   float_type delta_t, int nb_elements) {
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

template<typename float_type>
__global__  void t8gpu::timestepping::SSP_3RK_step2(float_type const* __restrict__ rho_prev,
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
						    float_type delta_t, int nb_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_elements) return;

  rho_2[i]    = rk_coeffs<float_type>::stage_2_1*rho_prev[i]    + rk_coeffs<float_type>::stage_2_2*rho_1[i]    + rk_coeffs<float_type>::stage_2_3*delta_t / volume[i] * rho_fluxes[i];
  rho_v1_2[i] = rk_coeffs<float_type>::stage_2_1*rho_v1_prev[i] + rk_coeffs<float_type>::stage_2_2*rho_v1_1[i] + rk_coeffs<float_type>::stage_2_3*delta_t / volume[i] * rho_v1_fluxes[i];
  rho_v2_2[i] = rk_coeffs<float_type>::stage_2_1*rho_v2_prev[i] + rk_coeffs<float_type>::stage_2_2*rho_v2_1[i] + rk_coeffs<float_type>::stage_2_3*delta_t / volume[i] * rho_v2_fluxes[i];
  rho_e_2[i]  = rk_coeffs<float_type>::stage_2_1*rho_e_prev[i]  + rk_coeffs<float_type>::stage_2_2*rho_e_1[i]  + rk_coeffs<float_type>::stage_2_3*delta_t / volume[i] * rho_e_fluxes[i];

  rho_fluxes[i]    = 0.0;
  rho_v1_fluxes[i] = 0.0;
  rho_v2_fluxes[i] = 0.0;
  rho_e_fluxes[i]  = 0.0;
}

template<typename float_type>
__global__  void t8gpu::timestepping::SSP_3RK_step3(float_type const* __restrict__ rho_prev,
						    float_type const* __restrict__ rho_v1_prev,
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
						    float_type delta_t, int nb_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_elements) return;

  rho_next[i]    = rk_coeffs<float_type>::stage_3_1*rho_prev[i]    + rk_coeffs<float_type>::stage_3_2*rho_2[i]    + rk_coeffs<float_type>::stage_3_3*delta_t / volume[i] * rho_fluxes[i];
  rho_v1_next[i] = rk_coeffs<float_type>::stage_3_1*rho_v1_prev[i] + rk_coeffs<float_type>::stage_3_2*rho_v1_2[i] + rk_coeffs<float_type>::stage_3_3*delta_t / volume[i] * rho_v1_fluxes[i];
  rho_v2_next[i] = rk_coeffs<float_type>::stage_3_1*rho_v2_prev[i] + rk_coeffs<float_type>::stage_3_2*rho_v2_2[i] + rk_coeffs<float_type>::stage_3_3*delta_t / volume[i] * rho_v2_fluxes[i];
  rho_e_next[i]  = rk_coeffs<float_type>::stage_3_1*rho_e_prev[i]  + rk_coeffs<float_type>::stage_3_2*rho_e_2[i]  + rk_coeffs<float_type>::stage_3_3*delta_t / volume[i] * rho_e_fluxes[i];

  rho_fluxes[i]    = 0.0;
  rho_v1_fluxes[i] = 0.0;
  rho_v2_fluxes[i] = 0.0;
  rho_e_fluxes[i]  = 0.0;
}
