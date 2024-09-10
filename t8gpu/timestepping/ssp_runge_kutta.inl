namespace t8gpu::timestepping {

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

}  // namespace t8gpu::timestepping

template<typename VariableType>
__global__ void t8gpu::timestepping::SSP_3RK_step1(
    t8gpu::MemoryAccessorOwn<VariableType> prev,
    t8gpu::MemoryAccessorOwn<VariableType> step1,
    t8gpu::MemoryAccessorOwn<VariableType> fluxes,
    typename t8gpu::variable_traits<VariableType>::float_type const* __restrict__ volume,
    typename t8gpu::variable_traits<VariableType>::float_type delta_t,
    int                                                       num_elements) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= num_elements) return;

  for (size_t k = 0; k < VariableType::nb_variables; k++) {
    step1.get(k)[i] = prev.get(k)[i] + delta_t / volume[i] * fluxes.get(k)[i];
  }

  for (size_t k = 0; k < VariableType::nb_variables; k++) {
    fluxes.get(k)[i] = 0.0;
  }
}

template<typename VariableType>
__global__ void t8gpu::timestepping::SSP_3RK_step2(
    t8gpu::MemoryAccessorOwn<VariableType> prev,
    t8gpu::MemoryAccessorOwn<VariableType> step1,
    t8gpu::MemoryAccessorOwn<VariableType> step2,
    t8gpu::MemoryAccessorOwn<VariableType> fluxes,
    typename t8gpu::variable_traits<VariableType>::float_type const* __restrict__ volume,
    typename t8gpu::variable_traits<VariableType>::float_type delta_t,
    int                                                       num_elements) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= num_elements) return;

  using float_type = typename variable_traits<VariableType>::float_type;
  for (size_t k = 0; k < VariableType::nb_variables; k++) {
    step2.get(k)[i] = rk_coeffs<float_type>::stage_2_1 * prev.get(k)[i] +
                      rk_coeffs<float_type>::stage_2_2 * step1.get(k)[i] +
                      rk_coeffs<float_type>::stage_2_3 * delta_t / volume[i] * fluxes.get(k)[i];
  }

  for (size_t k = 0; k < VariableType::nb_variables; k++) {
    fluxes.get(k)[i] = 0.0;
  }
}

template<typename VariableType>
__global__ void t8gpu::timestepping::SSP_3RK_step3(
    t8gpu::MemoryAccessorOwn<VariableType> prev,
    t8gpu::MemoryAccessorOwn<VariableType> step2,
    t8gpu::MemoryAccessorOwn<VariableType> next,
    t8gpu::MemoryAccessorOwn<VariableType> fluxes,
    typename t8gpu::variable_traits<VariableType>::float_type const* __restrict__ volume,
    typename t8gpu::variable_traits<VariableType>::float_type delta_t,
    int                                                       num_elements) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= num_elements) return;

  using float_type = typename variable_traits<VariableType>::float_type;
  for (size_t k = 0; k < VariableType::nb_variables; k++) {
    next.get(k)[i] = rk_coeffs<float_type>::stage_3_1 * prev.get(k)[i] +
                     rk_coeffs<float_type>::stage_3_2 * step2.get(k)[i] +
                     rk_coeffs<float_type>::stage_3_3 * delta_t / volume[i] * fluxes.get(k)[i];
  }

  for (size_t k = 0; k < VariableType::nb_variables; k++) {
    fluxes.get(k)[i] = 0.0;
  }
}

template<typename VariableType, typename SubgridType>
__global__ void t8gpu::timestepping::subgrid::SSP_3RK_step1(
    t8gpu::SubgridMemoryAccessorOwn<VariableType, SubgridType> prev,
    t8gpu::SubgridMemoryAccessorOwn<VariableType, SubgridType> step1,
    t8gpu::SubgridMemoryAccessorOwn<VariableType, SubgridType> fluxes,
    typename t8gpu::variable_traits<VariableType>::float_type const* __restrict__ volumes,
    typename t8gpu::variable_traits<VariableType>::float_type delta_t) {
  int const e_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;
  int const k = threadIdx.z;

  using float_type = typename variable_traits<VariableType>::float_type;
  float_type volume = volumes[e_idx] / static_cast<float_type>(SubgridType::size);

  for (size_t l = 0; l < VariableType::nb_variables; l++) {
    step1.get(l)(e_idx, i, j, k) = prev.get(l)(e_idx, i, j, k) + delta_t / volume * fluxes.get(l)(e_idx, i, j, k);
    fluxes.get(l)(e_idx, i, j, k) = 0.0;
  }
}

template<typename VariableType, typename SubgridType>
__global__ void t8gpu::timestepping::subgrid::SSP_3RK_step2(
    t8gpu::SubgridMemoryAccessorOwn<VariableType, SubgridType> prev,
    t8gpu::SubgridMemoryAccessorOwn<VariableType, SubgridType> step1,
    t8gpu::SubgridMemoryAccessorOwn<VariableType, SubgridType> step2,
    t8gpu::SubgridMemoryAccessorOwn<VariableType, SubgridType> fluxes,
    typename t8gpu::variable_traits<VariableType>::float_type const* __restrict__ volumes,
    typename t8gpu::variable_traits<VariableType>::float_type delta_t) {
  int const e_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;
  int const k = threadIdx.z;

  using float_type = typename variable_traits<VariableType>::float_type;
  float_type volume = volumes[e_idx] / static_cast<float_type>(SubgridType::size);

  using float_type = typename variable_traits<VariableType>::float_type;
  for (size_t l = 0; l < VariableType::nb_variables; l++) {
    step2.get(l)(e_idx, i, j, k) = rk_coeffs<float_type>::stage_2_1 * prev.get(l)(e_idx, i, j, k) +
                                   rk_coeffs<float_type>::stage_2_2 * step1.get(l)(e_idx, i, j, k) +
                                   rk_coeffs<float_type>::stage_2_3 * delta_t / volume * fluxes.get(l)(e_idx, i, j, k);
    fluxes.get(l)(e_idx, i, j, k) = 0.0;
  }
}

template<typename VariableType, typename SubgridType>
__global__ void t8gpu::timestepping::subgrid::SSP_3RK_step3(
    t8gpu::SubgridMemoryAccessorOwn<VariableType, SubgridType> prev,
    t8gpu::SubgridMemoryAccessorOwn<VariableType, SubgridType> step2,
    t8gpu::SubgridMemoryAccessorOwn<VariableType, SubgridType> next,
    t8gpu::SubgridMemoryAccessorOwn<VariableType, SubgridType> fluxes,
    typename t8gpu::variable_traits<VariableType>::float_type const* __restrict__ volumes,
    typename t8gpu::variable_traits<VariableType>::float_type delta_t) {
  int const e_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;
  int const k = threadIdx.z;

  using float_type = typename variable_traits<VariableType>::float_type;
  float_type volume = volumes[e_idx] / static_cast<float_type>(SubgridType::size);

  using float_type = typename variable_traits<VariableType>::float_type;
  for (size_t l = 0; l < VariableType::nb_variables; l++) {
    next.get(l)(e_idx, i, j, k) = rk_coeffs<float_type>::stage_3_1 * prev.get(l)(e_idx, i, j, k) +
                                  rk_coeffs<float_type>::stage_3_2 * step2.get(l)(e_idx, i, j, k) +
                                  rk_coeffs<float_type>::stage_3_3 * delta_t / volume * fluxes.get(l)(e_idx, i, j, k);
    fluxes.get(l)(e_idx, i, j, k) = 0.0;
  }
}
