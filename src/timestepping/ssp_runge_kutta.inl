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

template<typename float_type, size_t nb_variables>
__global__ void t8gpu::timestepping::SSP_3RK_step1(cuda::std::array<float_type* __restrict__, nb_variables> prev,
						   cuda::std::array<float_type* __restrict__, nb_variables> step1,
						   cuda::std::array<float_type* __restrict__, nb_variables> fluxes,
						   float_type const* __restrict__ volume,
						   float_type delta_t, int nb_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_elements) return;

  for (size_t k=0; k<nb_variables; k++) {
    step1[k][i] = prev[k][i] + delta_t / volume[i] * fluxes[k][i];
  }

  for (size_t k=0; k<nb_variables; k++) {
    fluxes[k][i] = 0.0;
  }
}

template<typename float_type, size_t nb_variables>
__global__ void t8gpu::timestepping::SSP_3RK_step2(cuda::std::array<float_type* __restrict__, nb_variables> prev,
						   cuda::std::array<float_type* __restrict__, nb_variables> step1,
						   cuda::std::array<float_type* __restrict__, nb_variables> step2,
						   cuda::std::array<float_type* __restrict__, nb_variables> fluxes,
						   float_type const* __restrict__ volume,
						   float_type delta_t, int nb_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_elements) return;

  for (size_t k=0; k<nb_variables; k++) {
    step2[k][i] = rk_coeffs<float_type>::stage_2_1*prev[k][i] + rk_coeffs<float_type>::stage_2_2*step1[k][i] + rk_coeffs<float_type>::stage_2_3*delta_t / volume[i] * fluxes[k][i];
  }

  for (size_t k=0; k<nb_variables; k++) {
    fluxes[k][i] = 0.0;
  }
}

template<typename float_type, size_t nb_variables>
__global__ void t8gpu::timestepping::SSP_3RK_step3(cuda::std::array<float_type* __restrict__, nb_variables> prev,
						   cuda::std::array<float_type* __restrict__, nb_variables> step2,
						   cuda::std::array<float_type* __restrict__, nb_variables> next,
						   cuda::std::array<float_type* __restrict__, nb_variables> fluxes,
						   float_type const* __restrict__ volume,
						   float_type delta_t, int nb_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_elements) return;

  for (size_t k=0; k<nb_variables; k++) {
    next[k][i] = rk_coeffs<float_type>::stage_3_1*prev[k][i] + rk_coeffs<float_type>::stage_3_2*step2[k][i] + rk_coeffs<float_type>::stage_3_3*delta_t / volume[i] * fluxes[k][i];
  }

  for (size_t k=0; k<nb_variables; k++) {
    fluxes[k][i] = 0.0;
  }
}
