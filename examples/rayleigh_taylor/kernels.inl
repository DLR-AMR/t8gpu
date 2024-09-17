
template<typename float_type>
struct numerical_constants {};

template<>
struct numerical_constants<float> {
  static constexpr float zero = 0.0f;
  static constexpr float half = 0.5f;
  static constexpr float one  = 1.0f;
};

template<>
struct numerical_constants<double> {
  static constexpr double zero = 0.0;
  static constexpr double half = 0.5;
  static constexpr double one  = 1.0;
};

using nc = numerical_constants<t8gpu::variable_traits<VariableList>::float_type>;

template<typename float_type>
__device__ static float_type ln_mean(float_type aL, float_type aR) {
  float_type Xi = aR / aL;
  float_type u  = (Xi * (Xi - float_type{2.0}) + float_type{1.0}) / (Xi * (Xi + float_type{2.0}) + float_type{1.0});

  float_type eps = float_type{1.0e-4};
  if (u < eps) {
    return (aL + aR) * float_type{52.50} /
           (float_type{105.0} + u * (float_type{35.0} + u * (float_type{21.0} + u * float_type{15.0})));
  } else {
    return (aR - aL) / log(Xi);
  }
}

template<typename float_type>
__device__ static void kepes_compute_flux(float_type  u_L[5],
                                          float_type  u_R[5],
                                          float_type  F_star[5],
                                          float_type& uHat,
                                          float_type& vHat,
                                          float_type& wHat,
                                          float_type& aHat,
                                          float_type& rhoHat,
                                          float_type& HHat,
                                          float_type& p1Hat) {
  float_type gamma    = float_type{1.4};  // TODO: remove this constant.
  float_type kappa    = gamma;
  float_type kappaM1  = kappa - nc::one;
  float_type sKappaM1 = nc::one / kappaM1;

  float_type sRho_L = nc::one / u_L[0];
  float_type velU_L = sRho_L * u_L[1];
  float_type velV_L = sRho_L * u_L[2];
  float_type velW_L = sRho_L * u_L[3];

  float_type sRho_R = nc::one / u_R[0];
  float_type velU_R = sRho_R * u_R[1];
  float_type velV_R = sRho_R * u_R[2];
  float_type velW_R = sRho_R * u_R[3];

  float_type Vel2s2_L = nc::half * (velU_L * velU_L + velV_L * velV_L + velW_L * velW_L);
  float_type Vel2s2_R = nc::half * (velU_R * velU_R + velV_R * velV_R + velW_R * velW_R);

  float_type p_L = kappaM1 * (u_L[4] - u_L[0] * Vel2s2_L);
  float_type p_R = kappaM1 * (u_R[4] - u_R[0] * Vel2s2_R);

  float_type beta_L = nc::half * u_L[0] / p_L;
  float_type beta_R = nc::half * u_R[0] / p_R;

  float_type rho_MEAN  = nc::half * (u_L[0] + u_R[0]);
  rhoHat               = ln_mean<float_type>(u_L[0], u_R[0]);
  float_type beta_MEAN = nc::half * (beta_L + beta_R);
  float_type beta_Hat  = ln_mean<float_type>(beta_L, beta_R);

  uHat  = nc::half * (velU_L + velU_R);
  vHat  = nc::half * (velV_L + velV_R);
  wHat  = nc::half * (velW_L + velW_R);
  aHat  = sqrt(kappa * nc::half * (p_L + p_R) / rhoHat);
  HHat  = kappa / (2.0f * kappaM1 * beta_Hat) + nc::half * (velU_L * velU_R + velV_L * velV_R + velW_L * velW_R);
  p1Hat = nc::half * rho_MEAN / beta_MEAN;
  float_type Vel2_M = Vel2s2_L + Vel2s2_R;

  float_type qHat = uHat;
  F_star[0]       = rhoHat * qHat;
  F_star[1]       = F_star[0] * uHat + p1Hat;
  F_star[2]       = F_star[0] * vHat;
  F_star[3]       = F_star[0] * wHat;
  F_star[4] =
      F_star[0] * nc::half * (sKappaM1 / beta_Hat - Vel2_M) + uHat * F_star[1] + vHat * F_star[2] + wHat * F_star[3];
}

template<typename float_type>
__device__ static void kepes_compute_diffusion_matrix(float_type  u_L[5],
                                                      float_type  u_R[5],
                                                      float_type  F_star[5],
                                                      float_type  RHat[5][5],
                                                      float_type  DHat[5],
                                                      float_type& uHat,
                                                      float_type& vHat,
                                                      float_type& wHat,
                                                      float_type& aHat,
                                                      float_type& rhoHat,
                                                      float_type& hHat,
                                                      float_type& p1Hat) {
  float_type gamma   = float_type{1.4};  // TODO: remove this constant.
  float_type kappa   = gamma;
  float_type kappaM1 = kappa - nc::one;

  kepes_compute_flux(u_L, u_R, F_star, uHat, vHat, wHat, aHat, rhoHat, hHat, p1Hat);

  float_type R_hat[5][5] = {
      {           nc::one,                    nc::one,nc::zero, nc::zero,nc::one                                                                                                    },
      {       uHat - aHat,                                                 uHat, nc::zero, nc::zero, uHat + aHat},
      {              vHat,                                                 vHat,  nc::one, nc::zero,        vHat},
      {              wHat,                                                 wHat, nc::zero,  nc::one,        wHat},
      {hHat - uHat * aHat,
       static_cast<float_type>(0.5) * (uHat * uHat + vHat * vHat + wHat * wHat),
       vHat,     wHat,
       hHat + uHat * aHat                                                                                       }
  };

  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < 5; j++) RHat[i][j] = R_hat[i][j];

  DHat[0] = nc::half * abs(uHat - aHat) * rhoHat / kappa;
  DHat[1] = abs(uHat) * (kappaM1 / kappa) * rhoHat;
  DHat[2] = abs(uHat) * p1Hat;
  DHat[3] = DHat[2];
  DHat[4] = nc::half * abs(uHat + aHat) * rhoHat / kappa;
}

template<typename float_type>
__device__ void complete_orthonormal_basis(float_type n[3], float_type t1[3], float_type t2[3]) {
  // We set the first tangential to be a vector different than the normal.
  t1[0] = n[1];
  t1[1] = n[2];
  t1[2] = -n[0];

  float_type dot_product = n[0] * t1[0] + n[1] * t1[1] + n[2] * t1[2];

  // We then project back to the tangential plane.
  t1[0] -= dot_product * n[0];
  t1[1] -= dot_product * n[1];
  t1[2] -= dot_product * n[2];

  float_type t1_norm = sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);

  t1[0] /= t1_norm;
  t1[1] /= t1_norm;
  t1[2] /= t1_norm;

  // The last basis vector is the cross product of the first two.
  t2[0] = n[1] * t1[2] - n[2] * t1[1];
  t2[1] = n[2] * t1[0] - n[0] * t1[2];
  t2[2] = n[0] * t1[1] - n[1] * t1[0];
}

template<typename float_type>
__device__ void rotate_state(
    float_type n[3], float_type t1[3], float_type t2[3], float_type state[5], float_type state_rotated[5]) {
  state_rotated[0] = state[0];
  state_rotated[1] = state[1] * n[0] + state[2] * n[1] + state[3] * n[2];
  state_rotated[2] = state[1] * t1[0] + state[2] * t1[1] + state[3] * t1[2];
  state_rotated[3] = state[1] * t2[0] + state[2] * t2[1] + state[3] * t2[2];
  state_rotated[4] = state[4];
}

template<typename float_type>
__device__ void inverse_rotate_state(
    float_type n[3], float_type t1[3], float_type t2[3], float_type state_rotated[5], float_type state[5]) {
  state[0] = state_rotated[0];
  state[1] = state_rotated[1] * n[0] + state_rotated[2] * t1[0] + state_rotated[3] * t2[0];
  state[2] = state_rotated[1] * n[1] + state_rotated[2] * t1[1] + state_rotated[3] * t2[1];
  state[3] = state_rotated[1] * n[2] + state_rotated[2] * t1[2] + state_rotated[3] * t2[2];
  state[4] = state_rotated[4];
}

template<typename float_type>
__device__ void compute_total_kepes_flux(float_type u_L[5], float_type u_R[5], float_type flux[5]) {
  float_type F_star[5];

  float_type RHat[5][5];
  float_type DHat[5];

  float_type uHat;
  float_type vHat;
  float_type wHat;
  float_type aHat;
  float_type rhoHat;
  float_type hHat;
  float_type p1Hat;
  kepes_compute_diffusion_matrix(u_L, u_R, F_star, RHat, DHat, uHat, vHat, wHat, aHat, rhoHat, hHat, p1Hat);

  // speed_estimates[i] = abs(uHat) + aHat;

  float_type kappa   = float_type{1.4};  // remove this constant
  float_type kappaM1 = kappa - nc::one;

  float_type sRho_L = nc::one / u_L[0];
  float_type sRho_R = nc::one / u_R[0];

  float_type Vel_L[3] = {u_L[1] * sRho_L, u_L[2] * sRho_L, u_L[3] * sRho_L};
  float_type Vel_R[3] = {u_R[1] * sRho_R, u_R[2] * sRho_R, u_R[3] * sRho_R};

  float_type p_L = kappaM1 * (u_L[4] - nc::half * (u_L[1] * Vel_L[0] + u_L[2] * Vel_L[1] + u_L[3] * Vel_L[2]));
  float_type p_R = kappaM1 * (u_R[4] - nc::half * (u_R[1] * Vel_R[0] + u_R[2] * Vel_R[1] + u_R[3] * Vel_R[2]));

  float_type sL = log(p_L) - kappa * log(u_L[0]);
  float_type sR = log(p_R) - kappa * log(u_R[0]);

  float_type rho_pL = u_L[0] / p_L;
  float_type rho_pR = u_R[0] / p_R;

  float_type vL[5];
  float_type vR[5];
  float_type vJump[5];
  float_type diss[5];

  vL[0] =
      (kappa - sL) / (kappaM1)-nc::half * rho_pL * (Vel_L[0] * Vel_L[0] + Vel_L[1] * Vel_L[1] + Vel_L[2] * Vel_L[2]);
  vR[0] =
      (kappa - sR) / (kappaM1)-nc::half * rho_pR * (Vel_R[0] * Vel_R[0] + Vel_R[1] * Vel_R[1] + Vel_R[2] * Vel_R[2]);

  vL[1] = rho_pL * Vel_L[0];
  vR[1] = rho_pR * Vel_R[0];

  vL[2] = rho_pL * Vel_L[1];
  vR[2] = rho_pR * Vel_R[1];

  vL[3] = rho_pL * Vel_L[2];
  vR[3] = rho_pR * Vel_R[2];

  vR[4] = -rho_pR;
  vL[4] = -rho_pL;

  for (size_t k = 0; k < 5; k++) {
    vJump[k] = vR[k] - vL[k];
  }
  for (size_t k = 0; k < 5; k++) {
    diss[k] = DHat[k] * (RHat[0][k] * vJump[0] + RHat[1][k] * vJump[1] + RHat[2][k] * vJump[2] + RHat[3][k] * vJump[3] +
                         RHat[4][k] * vJump[4]);
  }

  float_type diss_[5];
  for (size_t k = 0; k < 5; k++)
    diss_[k] = RHat[k][0] * diss[0] + RHat[k][1] * diss[1] + RHat[k][2] * diss[2] + RHat[k][3] * diss[3] +
               RHat[k][4] * diss[4];

  // Compute entropy stable numerical flux
  for (size_t k = 0; k < 5; k++) flux[k] = F_star[k] - nc::half * diss_[k];
}

template<typename SubgridType>
__global__ void compute_inner_fluxes(t8gpu::SubgridMemoryAccessorOwn<VariableList, SubgridType>              variables,
                                     t8gpu::SubgridMemoryAccessorOwn<VariableList, SubgridType>              fluxes,
                                     typename SubgridCompressibleEulerSolver<SubgridType>::float_type const* volumes) {
  using float_type = typename SubgridCompressibleEulerSolver<SubgridType>::float_type;

  int const e_idx = blockIdx.x;

  if constexpr (SubgridType::rank == 3) {
    int const i = threadIdx.x;
    int const j = threadIdx.y;
    int const k = threadIdx.z;

    auto [rho, rho_v1, rho_v2, rho_v3, rho_e] = variables.get(Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);
    auto [fluxes_rho, fluxes_rho_v1, fluxes_rho_v2, fluxes_rho_v3, fluxes_rho_e] =
        fluxes.get(Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);

    float_type volume      = volumes[e_idx];
    float_type edge_length = cbrt(volume) / static_cast<float_type>(SubgridType::template extent<0>);
    float_type surface     = edge_length * edge_length;

    __shared__ float_type shared_fluxes[VariableList::nb_variables * SubgridType::size];

    shared_fluxes[SubgridType::flat_index(i, j, k)]                         = 0.0;
    shared_fluxes[SubgridType::flat_index(i, j, k) + SubgridType::size]     = 0.0;
    shared_fluxes[SubgridType::flat_index(i, j, k) + 2 * SubgridType::size] = 0.0;
    shared_fluxes[SubgridType::flat_index(i, j, k) + 3 * SubgridType::size] = 0.0;
    shared_fluxes[SubgridType::flat_index(i, j, k) + 4 * SubgridType::size] = 0.0;

    if (i < SubgridType::template extent<0> - 1) {
      float_type n[3] = {1.0, 0.0, 0.0};
      float_type t1[3];
      float_type t2[3];

      complete_orthonormal_basis(n, t1, t2);

      float_type variables_left[5] = {rho(e_idx, i, j, k),
                                      rho_v1(e_idx, i, j, k),
                                      rho_v2(e_idx, i, j, k),
                                      rho_v3(e_idx, i, j, k),
                                      rho_e(e_idx, i, j, k)};

      float_type variables_right[5] = {rho(e_idx, i + 1, j, k),
                                       rho_v1(e_idx, i + 1, j, k),
                                       rho_v2(e_idx, i + 1, j, k),
                                       rho_v3(e_idx, i + 1, j, k),
                                       rho_e(e_idx, i + 1, j, k)};

      float_type variables_left_rotated[5];
      float_type variables_right_rotated[5];

      rotate_state(n, t1, t2, variables_left, variables_left_rotated);
      rotate_state(n, t1, t2, variables_right, variables_right_rotated);

      float_type flux_rotated[5];

      compute_total_kepes_flux(variables_left_rotated, variables_right_rotated, flux_rotated);

      float_type flux[5];

      inverse_rotate_state(n, t1, t2, flux_rotated, flux);

      shared_fluxes[SubgridType::flat_index(i, j, k)]                         = flux[0] * surface;
      shared_fluxes[SubgridType::flat_index(i, j, k) + SubgridType::size]     = flux[1] * surface;
      shared_fluxes[SubgridType::flat_index(i, j, k) + 2 * SubgridType::size] = flux[2] * surface;
      shared_fluxes[SubgridType::flat_index(i, j, k) + 3 * SubgridType::size] = flux[3] * surface;
      shared_fluxes[SubgridType::flat_index(i, j, k) + 4 * SubgridType::size] = flux[4] * surface;
    }
    __syncthreads();

    if (i < SubgridType::template extent<0> - 1) {
      fluxes_rho(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i, j, k)];
      fluxes_rho_v1(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i, j, k) + SubgridType::size];
      fluxes_rho_v2(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i, j, k) + 2 * SubgridType::size];
      fluxes_rho_v3(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i, j, k) + 3 * SubgridType::size];
      fluxes_rho_e(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i, j, k) + 4 * SubgridType::size];
    }

    if (i > 0) {
      fluxes_rho(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i - 1, j, k)];
      fluxes_rho_v1(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i - 1, j, k) + SubgridType::size];
      fluxes_rho_v2(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i - 1, j, k) + 2 * SubgridType::size];
      fluxes_rho_v3(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i - 1, j, k) + 3 * SubgridType::size];
      fluxes_rho_e(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i - 1, j, k) + 4 * SubgridType::size];
    }

    if (j < SubgridType::template extent<0> - 1) {
      float_type n[3] = {0.0, 1.0, 0.0};
      float_type t1[3];
      float_type t2[3];

      complete_orthonormal_basis(n, t1, t2);

      float_type variables_left[5] = {rho(e_idx, i, j, k),
                                      rho_v1(e_idx, i, j, k),
                                      rho_v2(e_idx, i, j, k),
                                      rho_v3(e_idx, i, j, k),
                                      rho_e(e_idx, i, j, k)};

      float_type variables_right[5] = {rho(e_idx, i, j + 1, k),
                                       rho_v1(e_idx, i, j + 1, k),
                                       rho_v2(e_idx, i, j + 1, k),
                                       rho_v3(e_idx, i, j + 1, k),
                                       rho_e(e_idx, i, j + 1, k)};

      float_type variables_left_rotated[5];
      float_type variables_right_rotated[5];

      rotate_state(n, t1, t2, variables_left, variables_left_rotated);
      rotate_state(n, t1, t2, variables_right, variables_right_rotated);

      float_type flux_rotated[5];

      compute_total_kepes_flux(variables_left_rotated, variables_right_rotated, flux_rotated);

      float_type flux[5];

      inverse_rotate_state(n, t1, t2, flux_rotated, flux);

      shared_fluxes[SubgridType::flat_index(i, j, k)]                         = flux[0] * surface;
      shared_fluxes[SubgridType::flat_index(i, j, k) + SubgridType::size]     = flux[1] * surface;
      shared_fluxes[SubgridType::flat_index(i, j, k) + 2 * SubgridType::size] = flux[2] * surface;
      shared_fluxes[SubgridType::flat_index(i, j, k) + 3 * SubgridType::size] = flux[3] * surface;
      shared_fluxes[SubgridType::flat_index(i, j, k) + 4 * SubgridType::size] = flux[4] * surface;
    }
    __syncthreads();

    if (j < SubgridType::template extent<0> - 1) {
      fluxes_rho(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i, j, k)];
      fluxes_rho_v1(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i, j, k) + SubgridType::size];
      fluxes_rho_v2(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i, j, k) + 2 * SubgridType::size];
      fluxes_rho_v3(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i, j, k) + 3 * SubgridType::size];
      fluxes_rho_e(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i, j, k) + 4 * SubgridType::size];
    }

    if (j > 0) {
      fluxes_rho(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i, j - 1, k)];
      fluxes_rho_v1(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i, j - 1, k) + SubgridType::size];
      fluxes_rho_v2(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i, j - 1, k) + 2 * SubgridType::size];
      fluxes_rho_v3(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i, j - 1, k) + 3 * SubgridType::size];
      fluxes_rho_e(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i, j - 1, k) + 4 * SubgridType::size];
    }

    if (k < SubgridType::template extent<0> - 1) {
      float_type n[3] = {0.0, 0.0, 1.0};
      float_type t1[3];
      float_type t2[3];

      complete_orthonormal_basis(n, t1, t2);

      float_type variables_left[5] = {rho(e_idx, i, j, k),
                                      rho_v1(e_idx, i, j, k),
                                      rho_v2(e_idx, i, j, k),
                                      rho_v3(e_idx, i, j, k),
                                      rho_e(e_idx, i, j, k)};

      float_type variables_right[5] = {rho(e_idx, i, j, k + 1),
                                       rho_v1(e_idx, i, j, k + 1),
                                       rho_v2(e_idx, i, j, k + 1),
                                       rho_v3(e_idx, i, j, k + 1),
                                       rho_e(e_idx, i, j, k + 1)};

      float_type variables_left_rotated[5];
      float_type variables_right_rotated[5];

      rotate_state(n, t1, t2, variables_left, variables_left_rotated);
      rotate_state(n, t1, t2, variables_right, variables_right_rotated);

      float_type flux_rotated[5];

      compute_total_kepes_flux(variables_left_rotated, variables_right_rotated, flux_rotated);

      float_type flux[5];

      inverse_rotate_state(n, t1, t2, flux_rotated, flux);

      shared_fluxes[SubgridType::flat_index(i, j, k)]                         = flux[0] * surface;
      shared_fluxes[SubgridType::flat_index(i, j, k) + SubgridType::size]     = flux[1] * surface;
      shared_fluxes[SubgridType::flat_index(i, j, k) + 2 * SubgridType::size] = flux[2] * surface;
      shared_fluxes[SubgridType::flat_index(i, j, k) + 3 * SubgridType::size] = flux[3] * surface;
      shared_fluxes[SubgridType::flat_index(i, j, k) + 4 * SubgridType::size] = flux[4] * surface;
    }
    __syncthreads();

    if (k < SubgridType::template extent<0> - 1) {
      fluxes_rho(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i, j, k)];
      fluxes_rho_v1(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i, j, k) + SubgridType::size];
      fluxes_rho_v2(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i, j, k) + 2 * SubgridType::size];
      fluxes_rho_v3(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i, j, k) + 3 * SubgridType::size];
      fluxes_rho_e(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i, j, k) + 4 * SubgridType::size];
    }

    if (k > 0) {
      fluxes_rho(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i, j, k - 1)];
      fluxes_rho_v1(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i, j, k - 1) + SubgridType::size];
      fluxes_rho_v2(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i, j, k - 1) + 2 * SubgridType::size];
      fluxes_rho_v3(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i, j, k - 1) + 3 * SubgridType::size];
      fluxes_rho_e(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i, j, k - 1) + 4 * SubgridType::size];
    }
  } else {
    int const i = threadIdx.x;
    int const j = threadIdx.y;

    auto [rho, rho_v1, rho_v2, rho_v3, rho_e] = variables.get(Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);
    auto [fluxes_rho, fluxes_rho_v1, fluxes_rho_v2, fluxes_rho_v3, fluxes_rho_e] =
        fluxes.get(Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);

    float_type volume      = volumes[e_idx];
    float_type edge_length = sqrt(volume) / static_cast<float_type>(SubgridType::template extent<0>);
    float_type surface     = edge_length;

    __shared__ float_type shared_fluxes[VariableList::nb_variables * SubgridType::size];

    shared_fluxes[SubgridType::flat_index(i, j)]                         = 0.0;
    shared_fluxes[SubgridType::flat_index(i, j) + SubgridType::size]     = 0.0;
    shared_fluxes[SubgridType::flat_index(i, j) + 2 * SubgridType::size] = 0.0;
    shared_fluxes[SubgridType::flat_index(i, j) + 3 * SubgridType::size] = 0.0;
    shared_fluxes[SubgridType::flat_index(i, j) + 4 * SubgridType::size] = 0.0;

    if (i < SubgridType::template extent<0> - 1) {
      float_type n[3] = {1.0, 0.0, 0.0};
      float_type t1[3];
      float_type t2[3];

      complete_orthonormal_basis(n, t1, t2);

      float_type variables_left[5] = {
          rho(e_idx, i, j), rho_v1(e_idx, i, j), rho_v2(e_idx, i, j), rho_v3(e_idx, i, j), rho_e(e_idx, i, j)};

      float_type variables_right[5] = {rho(e_idx, i + 1, j),
                                       rho_v1(e_idx, i + 1, j),
                                       rho_v2(e_idx, i + 1, j),
                                       rho_v3(e_idx, i + 1, j),
                                       rho_e(e_idx, i + 1, j)};

      float_type variables_left_rotated[5];
      float_type variables_right_rotated[5];

      rotate_state(n, t1, t2, variables_left, variables_left_rotated);
      rotate_state(n, t1, t2, variables_right, variables_right_rotated);

      float_type flux_rotated[5];

      compute_total_kepes_flux(variables_left_rotated, variables_right_rotated, flux_rotated);

      float_type flux[5];

      inverse_rotate_state(n, t1, t2, flux_rotated, flux);

      shared_fluxes[SubgridType::flat_index(i, j)]                         = flux[0] * surface;
      shared_fluxes[SubgridType::flat_index(i, j) + SubgridType::size]     = flux[1] * surface;
      shared_fluxes[SubgridType::flat_index(i, j) + 2 * SubgridType::size] = flux[2] * surface;
      shared_fluxes[SubgridType::flat_index(i, j) + 3 * SubgridType::size] = flux[3] * surface;
      shared_fluxes[SubgridType::flat_index(i, j) + 4 * SubgridType::size] = flux[4] * surface;
    }
    __syncthreads();

    if (i < SubgridType::template extent<0> - 1) {
      fluxes_rho(e_idx, i, j) -= shared_fluxes[SubgridType::flat_index(i, j)];
      fluxes_rho_v1(e_idx, i, j) -= shared_fluxes[SubgridType::flat_index(i, j) + SubgridType::size];
      fluxes_rho_v2(e_idx, i, j) -= shared_fluxes[SubgridType::flat_index(i, j) + 2 * SubgridType::size];
      fluxes_rho_v3(e_idx, i, j) -= shared_fluxes[SubgridType::flat_index(i, j) + 3 * SubgridType::size];
      fluxes_rho_e(e_idx, i, j) -= shared_fluxes[SubgridType::flat_index(i, j) + 4 * SubgridType::size];
    }

    if (i > 0) {
      fluxes_rho(e_idx, i, j) += shared_fluxes[SubgridType::flat_index(i - 1, j)];
      fluxes_rho_v1(e_idx, i, j) += shared_fluxes[SubgridType::flat_index(i - 1, j) + SubgridType::size];
      fluxes_rho_v2(e_idx, i, j) += shared_fluxes[SubgridType::flat_index(i - 1, j) + 2 * SubgridType::size];
      fluxes_rho_v3(e_idx, i, j) += shared_fluxes[SubgridType::flat_index(i - 1, j) + 3 * SubgridType::size];
      fluxes_rho_e(e_idx, i, j) += shared_fluxes[SubgridType::flat_index(i - 1, j) + 4 * SubgridType::size];
    }

    if (j < SubgridType::template extent<0> - 1) {
      float_type n[3] = {0.0, 1.0, 0.0};
      float_type t1[3];
      float_type t2[3];

      complete_orthonormal_basis(n, t1, t2);

      float_type variables_left[5] = {
          rho(e_idx, i, j), rho_v1(e_idx, i, j), rho_v2(e_idx, i, j), rho_v3(e_idx, i, j), rho_e(e_idx, i, j)};

      float_type variables_right[5] = {rho(e_idx, i, j + 1),
                                       rho_v1(e_idx, i, j + 1),
                                       rho_v2(e_idx, i, j + 1),
                                       rho_v3(e_idx, i, j + 1),
                                       rho_e(e_idx, i, j + 1)};

      float_type variables_left_rotated[5];
      float_type variables_right_rotated[5];

      rotate_state(n, t1, t2, variables_left, variables_left_rotated);
      rotate_state(n, t1, t2, variables_right, variables_right_rotated);

      float_type flux_rotated[5];

      compute_total_kepes_flux(variables_left_rotated, variables_right_rotated, flux_rotated);

      float_type flux[5];

      inverse_rotate_state(n, t1, t2, flux_rotated, flux);

      shared_fluxes[SubgridType::flat_index(i, j)]                         = flux[0] * surface;
      shared_fluxes[SubgridType::flat_index(i, j) + SubgridType::size]     = flux[1] * surface;
      shared_fluxes[SubgridType::flat_index(i, j) + 2 * SubgridType::size] = flux[2] * surface;
      shared_fluxes[SubgridType::flat_index(i, j) + 3 * SubgridType::size] = flux[3] * surface;
      shared_fluxes[SubgridType::flat_index(i, j) + 4 * SubgridType::size] = flux[4] * surface;
    }
    __syncthreads();

    if (j < SubgridType::template extent<0> - 1) {
      fluxes_rho(e_idx, i, j) -= shared_fluxes[SubgridType::flat_index(i, j)];
      fluxes_rho_v1(e_idx, i, j) -= shared_fluxes[SubgridType::flat_index(i, j) + SubgridType::size];
      fluxes_rho_v2(e_idx, i, j) -= shared_fluxes[SubgridType::flat_index(i, j) + 2 * SubgridType::size];
      fluxes_rho_v3(e_idx, i, j) -= shared_fluxes[SubgridType::flat_index(i, j) + 3 * SubgridType::size];
      fluxes_rho_e(e_idx, i, j) -= shared_fluxes[SubgridType::flat_index(i, j) + 4 * SubgridType::size];
    }

    if (j > 0) {
      fluxes_rho(e_idx, i, j) += shared_fluxes[SubgridType::flat_index(i, j - 1)];
      fluxes_rho_v1(e_idx, i, j) += shared_fluxes[SubgridType::flat_index(i, j - 1) + SubgridType::size];
      fluxes_rho_v2(e_idx, i, j) += shared_fluxes[SubgridType::flat_index(i, j - 1) + 2 * SubgridType::size];
      fluxes_rho_v3(e_idx, i, j) += shared_fluxes[SubgridType::flat_index(i, j - 1) + 3 * SubgridType::size];
      fluxes_rho_e(e_idx, i, j) += shared_fluxes[SubgridType::flat_index(i, j - 1) + 4 * SubgridType::size];
    }
  }
}

template<typename SubgridType>
__global__ void compute_outer_fluxes(
    typename t8gpu::SubgridMeshConnectivityAccessor<typename SubgridCompressibleEulerSolver<SubgridType>::float_type,
                                                    SubgridType> connectivity,
    t8gpu::SubgridMemoryAccessorAll<VariableList, SubgridType>   variables,
    t8gpu::SubgridMemoryAccessorAll<VariableList, SubgridType>   fluxes) {
  using subgrid_type = SubgridType;
  using float_type   = typename SubgridCompressibleEulerSolver<SubgridType>::float_type;

  int const f_idx = blockIdx.x;

  if constexpr (SubgridType::rank == 3) {
    int const i = threadIdx.x;
    int const j = threadIdx.y;

    t8_locidx_t level_difference = connectivity.get_face_level_difference(f_idx);

    int double_stride = (level_difference == 0) ? 2 : 1;

    std::array<int, 3> offset = connectivity.get_face_neighbor_offset(f_idx);

    float_type face_surface = connectivity.get_face_surface(f_idx);

    auto [l_idx, r_idx] = connectivity.get_face_neighbor_indices(f_idx);

    int l_rank  = connectivity.get_element_owner_rank(l_idx);
    int l_index = connectivity.get_element_owner_remote_index(l_idx);

    int r_rank  = connectivity.get_element_owner_rank(r_idx);
    int r_index = connectivity.get_element_owner_remote_index(r_idx);

    auto [nx, ny, nz] = connectivity.get_face_normal(f_idx);

    float_type n[3] = {nx, ny, nz};
    float_type t1[3];
    float_type t2[3];
    complete_orthonormal_basis(n, t1, t2);

    auto [rho_l, rho_v1_l, rho_v2_l, rho_v3_l, rho_e_l] = variables.get(l_rank, Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);
    auto [rho_r, rho_v1_r, rho_v2_r, rho_v3_r, rho_e_r] = variables.get(r_rank, Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);

    auto [flux_rho_l, flux_rho_v1_l, flux_rho_v2_l, flux_rho_v3_l, flux_rho_e_l] =
        fluxes.get(l_rank, Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);
    auto [flux_rho_r, flux_rho_v1_r, flux_rho_v2_r, flux_rho_v3_r, flux_rho_e_r] =
        fluxes.get(r_rank, Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);

    int anchor_l[3] = {0, 0, 0};

    int anchor_r[3] = {offset[0], offset[1], offset[2]};

    int stride_i[3] = {0, 0, 0};
    int stride_j[3] = {0, 0, 0};

    if (nx == 1.0) {
      anchor_l[0] = subgrid_type::template extent<0> - 1;

      stride_i[1] = 1;
      stride_j[2] = 1;
    }
    if (nx == -1.0) {
      stride_i[1] = 1;
      stride_j[2] = 1;
    }

    if (ny == 1.0) {
      anchor_l[1] = subgrid_type::template extent<1> - 1;

      stride_i[0] = 1;
      stride_j[2] = 1;
    }

    if (ny == -1.0) {
      stride_i[0] = 1;
      stride_j[2] = 1;
    }

    if (nz == 1.0) {
      anchor_l[2] = subgrid_type::template extent<2> - 1;

      stride_i[0] = 1;
      stride_j[1] = 1;
    }

    if (nz == -1.0) {
      stride_i[0] = 1;
      stride_j[1] = 1;
    }

    int l_i = anchor_l[0] + i * stride_i[0] + j * stride_j[0];
    int l_j = anchor_l[1] + i * stride_i[1] + j * stride_j[1];
    int l_k = anchor_l[2] + i * stride_i[2] + j * stride_j[2];

    int r_i = anchor_r[0] + double_stride * (i * stride_i[0] + j * stride_j[0]) / 2;
    int r_j = anchor_r[1] + double_stride * (i * stride_i[1] + j * stride_j[1]) / 2;
    int r_k = anchor_r[2] + double_stride * (i * stride_i[2] + j * stride_j[2]) / 2;

    float_type variables_left[5] = {rho_l(l_index, l_i, l_j, l_k),
                                    rho_v1_l(l_index, l_i, l_j, l_k),
                                    rho_v2_l(l_index, l_i, l_j, l_k),
                                    rho_v3_l(l_index, l_i, l_j, l_k),
                                    rho_e_l(l_index, l_i, l_j, l_k)};

    float_type variables_right[5] = {rho_r(r_index, r_i, r_j, r_k),
                                     rho_v1_r(r_index, r_i, r_j, r_k),
                                     rho_v2_r(r_index, r_i, r_j, r_k),
                                     rho_v3_r(r_index, r_i, r_j, r_k),
                                     rho_e_r(r_index, r_i, r_j, r_k)};

    float_type variables_left_rotated[5];
    float_type variables_right_rotated[5];

    rotate_state(n, t1, t2, variables_left, variables_left_rotated);
    rotate_state(n, t1, t2, variables_right, variables_right_rotated);

    float_type flux_rotated[5];

    compute_total_kepes_flux(variables_left_rotated, variables_right_rotated, flux_rotated);

    float_type flux[5];

    inverse_rotate_state(n, t1, t2, flux_rotated, flux);

    float_type surface =
        face_surface / static_cast<float_type>(subgrid_type::template extent<0> * subgrid_type::template extent<1>);

    atomicAdd(&flux_rho_l(l_index, l_i, l_j, l_k), -flux[0] * surface);
    atomicAdd(&flux_rho_r(r_index, r_i, r_j, r_k), flux[0] * surface);

    atomicAdd(&flux_rho_v1_l(l_index, l_i, l_j, l_k), -flux[1] * surface);
    atomicAdd(&flux_rho_v1_r(r_index, r_i, r_j, r_k), flux[1] * surface);

    atomicAdd(&flux_rho_v2_l(l_index, l_i, l_j, l_k), -flux[2] * surface);
    atomicAdd(&flux_rho_v2_r(r_index, r_i, r_j, r_k), flux[2] * surface);

    atomicAdd(&flux_rho_v3_l(l_index, l_i, l_j, l_k), -flux[3] * surface);
    atomicAdd(&flux_rho_v3_r(r_index, r_i, r_j, r_k), flux[3] * surface);

    atomicAdd(&flux_rho_e_l(l_index, l_i, l_j, l_k), -flux[4] * surface);
    atomicAdd(&flux_rho_e_r(r_index, r_i, r_j, r_k), flux[4] * surface);
  } else {
    int const i = threadIdx.x;

    t8_locidx_t level_difference = connectivity.get_face_level_difference(f_idx);

    int double_stride = (level_difference == 0) ? 2 : 1;

    std::array<int, 2> offset = connectivity.get_face_neighbor_offset(f_idx);

    float_type face_surface = connectivity.get_face_surface(f_idx);

    auto [l_idx, r_idx] = connectivity.get_face_neighbor_indices(f_idx);

    int l_rank  = connectivity.get_element_owner_rank(l_idx);
    int l_index = connectivity.get_element_owner_remote_index(l_idx);

    int r_rank  = connectivity.get_element_owner_rank(r_idx);
    int r_index = connectivity.get_element_owner_remote_index(r_idx);

    auto [nx, ny] = connectivity.get_face_normal(f_idx);

    float_type n[3] = {nx, ny, 0.0};
    float_type t1[3];
    float_type t2[3];
    complete_orthonormal_basis(n, t1, t2);

    auto [rho_l, rho_v1_l, rho_v2_l, rho_v3_l, rho_e_l] = variables.get(l_rank, Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);
    auto [rho_r, rho_v1_r, rho_v2_r, rho_v3_r, rho_e_r] = variables.get(r_rank, Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);

    auto [flux_rho_l, flux_rho_v1_l, flux_rho_v2_l, flux_rho_v3_l, flux_rho_e_l] =
        fluxes.get(l_rank, Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);
    auto [flux_rho_r, flux_rho_v1_r, flux_rho_v2_r, flux_rho_v3_r, flux_rho_e_r] =
        fluxes.get(r_rank, Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);

    int anchor_l[2] = {0, 0};

    int anchor_r[2] = {offset[0], offset[1]};

    int stride_i[2] = {0, 0};

    if (nx == 1.0) {
      anchor_l[0] = subgrid_type::template extent<0> - 1;

      stride_i[1] = 1;
    }
    if (nx == -1.0) {
      stride_i[1] = 1;
    }

    if (ny == 1.0) {
      anchor_l[1] = subgrid_type::template extent<1> - 1;

      stride_i[0] = 1;
    }

    if (ny == -1.0) {
      stride_i[0] = 1;
    }

    int l_i = anchor_l[0] + i * stride_i[0];
    int l_j = anchor_l[1] + i * stride_i[1];

    int r_i = anchor_r[0] + double_stride * (i * stride_i[0]) / 2;
    int r_j = anchor_r[1] + double_stride * (i * stride_i[1]) / 2;

    float_type variables_left[5] = {rho_l(l_index, l_i, l_j),
                                    rho_v1_l(l_index, l_i, l_j),
                                    rho_v2_l(l_index, l_i, l_j),
                                    rho_v3_l(l_index, l_i, l_j),
                                    rho_e_l(l_index, l_i, l_j)};

    float_type variables_right[5] = {rho_r(r_index, r_i, r_j),
                                     rho_v1_r(r_index, r_i, r_j),
                                     rho_v2_r(r_index, r_i, r_j),
                                     rho_v3_r(r_index, r_i, r_j),
                                     rho_e_r(r_index, r_i, r_j)};

    float_type variables_left_rotated[5];
    float_type variables_right_rotated[5];

    rotate_state(n, t1, t2, variables_left, variables_left_rotated);
    rotate_state(n, t1, t2, variables_right, variables_right_rotated);

    float_type flux_rotated[5];

    compute_total_kepes_flux(variables_left_rotated, variables_right_rotated, flux_rotated);

    float_type flux[5];

    inverse_rotate_state(n, t1, t2, flux_rotated, flux);

    float_type surface = face_surface / static_cast<float_type>(subgrid_type::template extent<0>);

    atomicAdd(&flux_rho_l(l_index, l_i, l_j), -flux[0] * surface);
    atomicAdd(&flux_rho_r(r_index, r_i, r_j), flux[0] * surface);

    atomicAdd(&flux_rho_v1_l(l_index, l_i, l_j), -flux[1] * surface);
    atomicAdd(&flux_rho_v1_r(r_index, r_i, r_j), flux[1] * surface);

    atomicAdd(&flux_rho_v2_l(l_index, l_i, l_j), -flux[2] * surface);
    atomicAdd(&flux_rho_v2_r(r_index, r_i, r_j), flux[2] * surface);

    atomicAdd(&flux_rho_v3_l(l_index, l_i, l_j), -flux[3] * surface);
    atomicAdd(&flux_rho_v3_r(r_index, r_i, r_j), flux[3] * surface);

    atomicAdd(&flux_rho_e_l(l_index, l_i, l_j), -flux[4] * surface);
    atomicAdd(&flux_rho_e_r(r_index, r_i, r_j), flux[4] * surface);
  }
}

template<typename SubgridType>
__global__ void compute_refinement_criteria(
    typename SubgridType::Accessor<typename SubgridCompressibleEulerSolver<SubgridType>::float_type> density,
    typename SubgridCompressibleEulerSolver<SubgridType>::float_type*       refinement_criteria,
    typename SubgridCompressibleEulerSolver<SubgridType>::float_type const* volumes,
    t8_locidx_t                                                             num_local_elements) {
  using float_type = typename SubgridCompressibleEulerSolver<SubgridType>::float_type;

  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_local_elements) return;

  float_type h1_seminorm = float_type{0.0};

  if constexpr (SubgridType::rank == 3) {
    float_type h = cbrt(volumes[i]) / static_cast<float_type>(SubgridType::template extent<0>);

    for (size_t p = 0; p < SubgridType::template extent<0> - 1; p++) {
      for (size_t q = 0; q < SubgridType::template extent<1>; q++) {
        for (size_t r = 0; r < SubgridType::template extent<2>; r++) {
          h1_seminorm +=
              (density(i, p + 1, q, r) - density(i, p, q, r)) * (density(i, p + 1, q, r) - density(i, p, q, r)) * h;
        }
      }
    }

    for (size_t p = 0; p < SubgridType::template extent<0>; p++) {
      for (size_t q = 0; q < SubgridType::template extent<1> - 1; q++) {
        for (size_t r = 0; r < SubgridType::template extent<2>; r++) {
          h1_seminorm +=
              (density(i, p, q + 1, r) - density(i, p, q, r)) * (density(i, p, q + 1, r) - density(i, p, q, r)) * h;
        }
      }
    }

    for (size_t p = 0; p < SubgridType::template extent<0>; p++) {
      for (size_t q = 0; q < SubgridType::template extent<1>; q++) {
        for (size_t r = 0; r < SubgridType::template extent<2> - 1; r++) {
          h1_seminorm +=
              (density(i, p, q, r + 1) - density(i, p, q, r)) * (density(i, p, q, r + 1) - density(i, p, q, r)) * h;
        }
      }
    }
  } else {
    float_type h = sqrt(volumes[i]) / static_cast<float_type>(SubgridType::template extent<0>);

    for (size_t p = 0; p < SubgridType::template extent<0> - 1; p++) {
      for (size_t q = 0; q < SubgridType::template extent<1>; q++) {
        h1_seminorm += (density(i, p + 1, q) - density(i, p, q)) * (density(i, p + 1, q) - density(i, p, q)) * h;
      }
    }

    for (size_t p = 0; p < SubgridType::template extent<0>; p++) {
      for (size_t q = 0; q < SubgridType::template extent<1> - 1; q++) {
        h1_seminorm += (density(i, p, q + 1) - density(i, p, q)) * (density(i, p, q + 1) - density(i, p, q)) * h;
      }
    }
  }

  refinement_criteria[i] = h1_seminorm / volumes[i];
}
