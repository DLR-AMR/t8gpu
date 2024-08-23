#include <solvers/compressible_euler/kernels.h>

using namespace t8gpu;

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

using nc = numerical_constants<variable_traits<VariableList>::float_type>;

template<typename float_type>
__device__ static float_type ln_mean(float_type aL, float_type aR) {
  float_type Xi = aR/aL;
  float_type u = (Xi*(Xi-float_type{2.0})+float_type{1.0})/(Xi*(Xi+float_type{2.0})+float_type{1.0});

  float_type eps = float_type{1.0e-4};
  if (u < eps) {
    return (aL+aR)*float_type{52.50}/(float_type{105.0} + u*(float_type{35.0} + u*(float_type{21.0} + u*float_type{15.0})));
  } else {
    return (aR-aL)/log(Xi);
  }
}

template<typename float_type>
__device__ static void kepes_compute_flux(float_type u_L[5],
					  float_type u_R[5],
					  float_type F_star[5],
					  float_type& uHat,
					  float_type& vHat,
					  float_type& wHat,
					  float_type& aHat,
					  float_type& rhoHat,
					  float_type& HHat,
					  float_type& p1Hat) {
  float_type gamma = float_type{1.4}; // TODO: remove this constant.
  float_type kappa = gamma;
  float_type kappaM1 = kappa - nc::one;
  float_type sKappaM1 = nc::one/kappaM1;

  float_type sRho_L = nc::one/u_L[0];
  float_type velU_L = sRho_L*u_L[1];
  float_type velV_L = sRho_L*u_L[2];
  float_type velW_L = sRho_L*u_L[3];

  float_type sRho_R = nc::one/u_R[0];
  float_type velU_R = sRho_R*u_R[1];
  float_type velV_R = sRho_R*u_R[2];
  float_type velW_R = sRho_R*u_R[3];

  float_type Vel2s2_L = nc::half*(velU_L*velU_L+velV_L*velV_L+velW_L*velW_L);
  float_type Vel2s2_R = nc::half*(velU_R*velU_R+velV_R*velV_R+velW_R*velW_R);

  float_type p_L = kappaM1*(u_L[4] - u_L[0]*Vel2s2_L);
  float_type p_R = kappaM1*(u_R[4] - u_R[0]*Vel2s2_R);

  float_type beta_L = nc::half*u_L[0]/p_L;
  float_type beta_R = nc::half*u_R[0]/p_R;

  float_type rho_MEAN  = nc::half*(u_L[0]+u_R[0]);
  rhoHat    = ln_mean<float_type>(u_L[0],u_R[0]);
  float_type beta_MEAN = nc::half*(beta_L+beta_R);
  float_type beta_Hat  = ln_mean<float_type>(beta_L,beta_R);

  uHat      = nc::half*(velU_L+velU_R);
  vHat      = nc::half*(velV_L+velV_R);
  wHat      = nc::half*(velW_L+velW_R);
  aHat      = sqrt(kappa*nc::half*(p_L+p_R)/rhoHat);
  HHat      = kappa/(2.0f*kappaM1*beta_Hat) + nc::half*(velU_L*velU_R+velV_L*velV_R+velW_L*velW_R);
  p1Hat     = nc::half*rho_MEAN/beta_MEAN;
  float_type Vel2_M    = Vel2s2_L+Vel2s2_R;

  float_type qHat      = uHat;
  F_star[0] = rhoHat*qHat;
  F_star[1] = F_star[0]*uHat + p1Hat;
  F_star[2] = F_star[0]*vHat;
  F_star[3] = F_star[0]*wHat;
  F_star[4] = F_star[0]*nc::half*(sKappaM1/beta_Hat - Vel2_M) + uHat*F_star[1] + vHat*F_star[2] + wHat*F_star[3];
}

template<typename float_type>
__device__ static void kepes_compute_diffusion_matrix(float_type u_L[5],
						      float_type u_R[5],
						      float_type F_star[5],
						      float_type RHat[5][5],
						      float_type DHat[5],
						      float_type& uHat,
						      float_type& vHat,
						      float_type& wHat,
						      float_type& aHat,
						      float_type& rhoHat,
						      float_type& hHat,
						      float_type& p1Hat) {
  float_type gamma = float_type{1.4}; // TODO: remove this constant.
  float_type kappa = gamma;
  float_type kappaM1 = kappa - nc::one;

  kepes_compute_flux(u_L,
		     u_R,
		     F_star,
		     uHat,
		     vHat,
		     wHat,
		     aHat,
		     rhoHat,
		     hHat,
		     p1Hat);

  float_type R_hat[5][5] = {
    {       nc::one,                                                          nc::one, nc::zero, nc::zero,        nc::one},
    {     uHat-aHat,                                                             uHat, nc::zero, nc::zero,      uHat+aHat},
    {          vHat,                                                             vHat,  nc::one, nc::zero,           vHat},
    {          wHat,                                                             wHat, nc::zero,  nc::one,           wHat},
    {hHat-uHat*aHat, static_cast<float_type>(0.5)*(uHat*uHat + vHat*vHat + wHat*wHat),     vHat,     wHat, hHat+uHat*aHat}
  };

  for (size_t i=0; i<5; i++)
    for (size_t j=0; j<5; j++)
      RHat[i][j] = R_hat[i][j];

  DHat[0] = nc::half*abs(uHat-aHat)*rhoHat/kappa;
  DHat[1] = abs(uHat)*(kappaM1/kappa)*rhoHat;
  DHat[2] = abs(uHat)*p1Hat;
  DHat[3] = DHat[2];
  DHat[4] = nc::half*abs(uHat+aHat)*rhoHat/kappa;

}

__global__ void t8gpu::kepes_compute_fluxes(MeshConnectivityAccessor<typename variable_traits<VariableList>::float_type, CompressibleEulerSolver::dim> connectivity,
					    MemoryAccessorAll<VariableList> variables,
					    MemoryAccessorAll<VariableList> fluxes,
					    typename variable_traits<VariableList>::float_type* __restrict__ speed_estimates) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_local_faces = connectivity.get_num_local_faces();
  if (i >= num_local_faces) return;

  using float_type = typename variable_traits<VariableList>::float_type;

  // float_type face_surface = area[i];
  float_type face_surface = connectivity.get_face_surface(i);

  auto [l_idx, r_idx] = connectivity.get_face_neighbor_indices(i);

  int l_rank  = connectivity.get_element_owner_rank(l_idx);
  int l_index = connectivity.get_element_owner_remote_index(l_idx);

  int r_rank  = connectivity.get_element_owner_rank(r_idx);
  int r_index = connectivity.get_element_owner_remote_index(r_idx);

  auto [nx, ny, nz] = connectivity.get_face_normal(i);

  auto [rho, rho_v1, rho_v2, rho_v3, rho_e] = variables.get(Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);

  float_type rho_l    = rho[l_rank][l_index];
  float_type rho_vx_l = rho_v1[l_rank][l_index];
  float_type rho_vy_l = rho_v2[l_rank][l_index];
  float_type rho_vz_l = rho_v3[l_rank][l_index];
  float_type rho_e_l  = rho_e[l_rank][l_index];

  float_type rho_r    = rho[r_rank][r_index];
  float_type rho_vx_r = rho_v1[r_rank][r_index];
  float_type rho_vy_r = rho_v2[r_rank][r_index];
  float_type rho_vz_r = rho_v3[r_rank][r_index];
  float_type rho_e_r  = rho_e[r_rank][r_index];

  float_type n[3] = {nx, ny, nz};

  // We set the first tangential to be a vector different than the normal.
  float_type t1[3] = {ny, nz, -nx};

  float_type dot_product = n[0]*t1[0] + n[1]*t1[1] + n[2]*t1[2];

  // We then project back to the tangential plane.
  t1[0] -= dot_product*n[0];
  t1[1] -= dot_product*n[1];
  t1[2] -= dot_product*n[2];

  // The last basis vector is the cross product of the first two.
  float_type t2[3] = {
    n[1]*t1[2]-n[2]*t1[1],
    n[2]*t1[0]-n[0]*t1[2],
    n[0]*t1[1]-n[1]*t1[0]
  };

  // rotate from (x,y,z) basis to local basis (n,t1,t2)
  float_type u_L[5] = {
    rho_l,
    rho_vx_l*n[0] + rho_vy_l*n[1] + rho_vz_l*n[2],
    rho_vx_l*t1[0] + rho_vy_l*t1[1] + rho_vz_l*t1[2],
    rho_vx_l*t2[0] + rho_vy_l*t2[1] + rho_vz_l*t2[2],
    rho_e_l
  };

  float_type u_R[5] = {
    rho_r,
    rho_vx_r*n[0] + rho_vy_r*n[1] + rho_vz_r*n[2],
    rho_vx_r*t1[0] + rho_vy_r*t1[1] + rho_vz_r*t1[2],
    rho_vx_r*t2[0] + rho_vy_r*t2[1] + rho_vz_r*t2[2],
    rho_e_r
  };

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
  kepes_compute_diffusion_matrix(u_L,
				 u_R,
				 F_star,
				 RHat,
				 DHat,
				 uHat,
				 vHat,
				 wHat,
				 aHat,
				 rhoHat,
				 hHat,
				 p1Hat);

  speed_estimates[i] = abs(uHat) + aHat;


  float_type kappa = float_type{1.4}; // remove this constant
  float_type kappaM1 = kappa - nc::one;

  float_type sRho_L = nc::one/u_L[0];
  float_type sRho_R = nc::one/u_R[0];

  float_type Vel_L[3] = {u_L[1]*sRho_L, u_L[2]*sRho_L, u_L[3]*sRho_L};
  float_type Vel_R[3] = {u_R[1]*sRho_R, u_R[2]*sRho_R, u_R[3]*sRho_R};

  float_type p_L = kappaM1*(u_L[4]-nc::half*(u_L[1]*Vel_L[0] + u_L[2]*Vel_L[1] + u_L[3]*Vel_L[2]));
  float_type p_R = kappaM1*(u_R[4]-nc::half*(u_R[1]*Vel_R[0] + u_R[2]*Vel_R[1] + u_R[3]*Vel_R[2]));

  float_type sL =  log(p_L) - kappa*log(u_L[0]);
  float_type sR =  log(p_R) - kappa*log(u_R[0]);

  float_type rho_pL = u_L[0]/p_L;
  float_type rho_pR = u_R[0]/p_R;

  float_type vL[5];
  float_type vR[5];
  float_type vJump[5];
  float_type diss[5];

  vL[0] =  (kappa-sL)/(kappaM1) - nc::half*rho_pL*(Vel_L[0]*Vel_L[0] + Vel_L[1]*Vel_L[1] + Vel_L[2]*Vel_L[2]);
  vR[0] =  (kappa-sR)/(kappaM1) - nc::half*rho_pR*(Vel_R[0]*Vel_R[0] + Vel_R[1]*Vel_R[1] + Vel_R[2]*Vel_R[2]);

  vL[1] = rho_pL*Vel_L[0];
  vR[1] = rho_pR*Vel_R[0];

  vL[2] = rho_pL*Vel_L[1];
  vR[2] = rho_pR*Vel_R[1];

  vL[3] = rho_pL*Vel_L[2];
  vR[3] = rho_pR*Vel_R[2];

  vR[4] = -rho_pR;
  vL[4] = -rho_pL;

  for (size_t k=0; k<5; k++) {
    vJump[k] = vR[k] - vL[k];
  }
  for (size_t k=0; k<5; k++) {
    diss[k]  = DHat[k]*(RHat[0][k]*vJump[0] + RHat[1][k]*vJump[1] + RHat[2][k]*vJump[2] + RHat[3][k]*vJump[3] + RHat[4][k]*vJump[4]);
  }

  float_type diss_[5];
  for (size_t k=0; k<5; k++)
    diss_[k] = RHat[k][0]*diss[0] + RHat[k][1]*diss[1] + RHat[k][2]*diss[2] + RHat[k][3]*diss[3] + RHat[k][4]*diss[4];

  // Compute entropy stable numerical flux
  float_type F[5];
  for (size_t k=0; k<5; k++)
    F[k] = F_star[k] - nc::half*diss_[k];

  float_type rho_flux    = face_surface*F[0];
  float_type rho_v1_flux = face_surface*F[1];
  float_type rho_v2_flux = face_surface*F[2];
  float_type rho_v3_flux = face_surface*F[3];
  float_type rho_e_flux  = face_surface*F[4];

  // rotate back from (n,t1,t2) to (x,y,z) basis.
  float_type rho_vx_flux = rho_v1_flux*n[0] + rho_v2_flux*t1[0] * rho_v3_flux*t2[0];
  float_type rho_vy_flux = rho_v1_flux*n[1] + rho_v2_flux*t1[1] * rho_v3_flux*t2[1];
  float_type rho_vz_flux = rho_v1_flux*n[2] + rho_v2_flux*t1[2] * rho_v3_flux*t2[2];

  auto [fluxes_rho, fluxes_rho_v1, fluxes_rho_v2, fluxes_rho_v3, fluxes_rho_e] = fluxes.get(Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);

  atomicAdd(&fluxes_rho[l_rank][l_index], -rho_flux);
  atomicAdd(&fluxes_rho[r_rank][r_index],  rho_flux);

  atomicAdd(&fluxes_rho_v1[l_rank][l_index], -rho_vx_flux);
  atomicAdd(&fluxes_rho_v1[r_rank][r_index],  rho_vx_flux);

  atomicAdd(&fluxes_rho_v2[l_rank][l_index], -rho_vy_flux);
  atomicAdd(&fluxes_rho_v2[r_rank][r_index],  rho_vy_flux);

  atomicAdd(&fluxes_rho_v3[l_rank][l_index], -rho_vz_flux);
  atomicAdd(&fluxes_rho_v3[r_rank][r_index],  rho_vz_flux);

  atomicAdd(&fluxes_rho_e[l_rank][l_index], -rho_e_flux);
  atomicAdd(&fluxes_rho_e[r_rank][r_index],  rho_e_flux);
}

__global__ void t8gpu::reflective_boundary_condition(MeshConnectivityAccessor<typename variable_traits<VariableList>::float_type, CompressibleEulerSolver::dim> connectivity,
						     MemoryAccessorOwn<VariableList> variables,
						     MemoryAccessorOwn<VariableList> fluxes,
						     typename variable_traits<VariableList>::float_type* __restrict__ speed_estimates) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_local_faces = connectivity.get_num_local_boundary_faces();
  if (i >= num_local_faces) return;

  using float_type = typename variable_traits<VariableList>::float_type;

  float_type face_surface = connectivity.get_boundary_face_surface(i);

  auto e_idx = connectivity.get_boundary_face_neighbor_index(i);

  auto [nx, ny, nz] = connectivity.get_boundary_face_normal(i);

  auto [rho, rho_v1, rho_v2, rho_v3, rho_e] = variables.get(Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);

  float_type rho_l    = rho[e_idx];
  float_type rho_vx_l = rho_v1[e_idx];
  float_type rho_vy_l = rho_v2[e_idx];
  float_type rho_vz_l = rho_v3[e_idx];
  float_type rho_e_l  = rho_e[e_idx];

  float_type rho_r    = rho_l;
  float_type rho_vx_r = rho_vx_l;
  float_type rho_vy_r = rho_vy_l;
  float_type rho_vz_r = rho_vz_l;
  float_type rho_e_r  = rho_e_l;

  float_type n[3] = {nx, ny, nz};

  // We set the first tangential to be a vector different than the normal.
  float_type t1[3] = {ny, nz, -nx};

  float_type dot_product = n[0]*t1[0] + n[1]*t1[1] + n[2]*t1[2];

  // We then project back to the tangential plane.
  t1[0] -= dot_product*n[0];
  t1[1] -= dot_product*n[1];
  t1[2] -= dot_product*n[2];

  // The last basis vector is the cross product of the first two.
  float_type t2[3] = {
    n[1]*t1[2]-n[2]*t1[1],
    n[2]*t1[0]-n[0]*t1[2],
    n[0]*t1[1]-n[1]*t1[0]
  };

  // rotate from (x,y,z) basis to local basis (n,t1,t2)
  float_type u_L[5] = {
    rho_l,
    rho_vx_l*n[0] + rho_vy_l*n[1] + rho_vz_l*n[2],
    rho_vx_l*t1[0] + rho_vy_l*t1[1] + rho_vz_l*t1[2],
    rho_vx_l*t2[0] + rho_vy_l*t2[1] + rho_vz_l*t2[2],
    rho_e_l
  };

  float_type u_R[5] = {
    rho_r,
    -(rho_vx_r*n[0] + rho_vy_r*n[1] + rho_vz_r*n[2]),
    rho_vx_r*t1[0] + rho_vy_r*t1[1] + rho_vz_r*t1[2],
    rho_vx_r*t2[0] + rho_vy_r*t2[1] + rho_vz_r*t2[2],
    rho_e_r
  };

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
  kepes_compute_diffusion_matrix(u_L,
				 u_R,
				 F_star,
				 RHat,
				 DHat,
				 uHat,
				 vHat,
				 wHat,
				 aHat,
				 rhoHat,
				 hHat,
				 p1Hat);

  speed_estimates[i] = abs(uHat) + aHat;


  float_type kappa = float_type{1.4}; // remove this constant
  float_type kappaM1 = kappa - nc::one;

  float_type sRho_L = nc::one/u_L[0];
  float_type sRho_R = nc::one/u_R[0];

  float_type Vel_L[3] = {u_L[1]*sRho_L, u_L[2]*sRho_L, u_L[3]*sRho_L};
  float_type Vel_R[3] = {u_R[1]*sRho_R, u_R[2]*sRho_R, u_R[3]*sRho_R};

  float_type p_L = kappaM1*(u_L[4]-nc::half*(u_L[1]*Vel_L[0] + u_L[2]*Vel_L[1] + u_L[3]*Vel_L[2]));
  float_type p_R = kappaM1*(u_R[4]-nc::half*(u_R[1]*Vel_R[0] + u_R[2]*Vel_R[1] + u_R[3]*Vel_R[2]));

  float_type sL =  log(p_L) - kappa*log(u_L[0]);
  float_type sR =  log(p_R) - kappa*log(u_R[0]);

  float_type rho_pL = u_L[0]/p_L;
  float_type rho_pR = u_R[0]/p_R;

  float_type vL[5];
  float_type vR[5];
  float_type vJump[5];
  float_type diss[5];

  vL[0] =  (kappa-sL)/(kappaM1) - nc::half*rho_pL*(Vel_L[0]*Vel_L[0] + Vel_L[1]*Vel_L[1] + Vel_L[2]*Vel_L[2]);
  vR[0] =  (kappa-sR)/(kappaM1) - nc::half*rho_pR*(Vel_R[0]*Vel_R[0] + Vel_R[1]*Vel_R[1] + Vel_R[2]*Vel_R[2]);

  vL[1] = rho_pL*Vel_L[0];
  vR[1] = rho_pR*Vel_R[0];

  vL[2] = rho_pL*Vel_L[1];
  vR[2] = rho_pR*Vel_R[1];

  vL[3] = rho_pL*Vel_L[2];
  vR[3] = rho_pR*Vel_R[2];

  vR[4] = -rho_pR;
  vL[4] = -rho_pL;

  for (size_t k=0; k<5; k++) {
    vJump[k] = vR[k] - vL[k];
  }
  for (size_t k=0; k<5; k++) {
    diss[k]  = DHat[k]*(RHat[0][k]*vJump[0] + RHat[1][k]*vJump[1] + RHat[2][k]*vJump[2] + RHat[3][k]*vJump[3] + RHat[4][k]*vJump[4]);
  }

  float_type diss_[5];
  for (size_t k=0; k<5; k++)
    diss_[k] = RHat[k][0]*diss[0] + RHat[k][1]*diss[1] + RHat[k][2]*diss[2] + RHat[k][3]*diss[3] + RHat[k][4]*diss[4];

  // Compute entropy stable numerical flux
  float_type F[5];
  for (size_t k=0; k<5; k++)
    F[k] = F_star[k] - nc::half*diss_[k];

  float_type rho_flux    = face_surface*F[0];
  float_type rho_v1_flux = face_surface*F[1];
  float_type rho_v2_flux = face_surface*F[2];
  float_type rho_v3_flux = face_surface*F[3];
  float_type rho_e_flux  = face_surface*F[4];

  // rotate back from (n,t1,t2) to (x,y,z) basis.
  float_type rho_vx_flux = rho_v1_flux*n[0] + rho_v2_flux*t1[0] * rho_v3_flux*t2[0];
  float_type rho_vy_flux = rho_v1_flux*n[1] + rho_v2_flux*t1[1] * rho_v3_flux*t2[1];
  float_type rho_vz_flux = rho_v1_flux*n[2] + rho_v2_flux*t1[2] * rho_v3_flux*t2[2];

  auto [fluxes_rho, fluxes_rho_v1, fluxes_rho_v2, fluxes_rho_v3, fluxes_rho_e] = fluxes.get(Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);

  atomicAdd(&fluxes_rho[e_idx], -rho_flux);
  atomicAdd(&fluxes_rho_v1[e_idx], -rho_vx_flux);
  atomicAdd(&fluxes_rho_v2[e_idx], -rho_vy_flux);
  atomicAdd(&fluxes_rho_v3[e_idx], -rho_vz_flux);
  atomicAdd(&fluxes_rho_e[e_idx], -rho_e_flux);
}

__global__ void t8gpu::estimate_gradient(MeshConnectivityAccessor<typename variable_traits<VariableList>::float_type, 3> connectivity,
					 MemoryAccessorAll<VariableList> data_next,
					 MemoryAccessorAll<VariableList> data_fluxes) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_local_faces = connectivity.get_num_local_faces();
  if (i >= num_local_faces) return;

  using float_type = typename variable_traits<VariableList>::float_type;

  auto [rho, rho_v1] = data_next.get(Rho, Rho_v1);
  auto rho_bis = data_next.get(Rho);

  auto rho_gradient = data_fluxes.get(Rho);

  auto [l_idx, r_idx] = connectivity.get_face_neighbor_indices(i);

  int l_rank  = connectivity.get_element_owner_rank(l_idx);
  int l_index = connectivity.get_element_owner_remote_index(l_idx);

  int r_rank  = connectivity.get_element_owner_rank(r_idx);
  int r_index = connectivity.get_element_owner_remote_index(r_idx);

  float_type rho_l = rho[l_rank][l_index];
  float_type rho_r = rho[r_rank][r_index];

  float_type gradient = abs(rho_r - rho_l);

  atomicAdd(&rho_gradient[l_rank][l_index], gradient);
  atomicAdd(&rho_gradient[r_rank][r_index], gradient);
}
