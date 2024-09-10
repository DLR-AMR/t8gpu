#include "kernels.h"

using namespace t8gpu;

#define NX (1.0)
#define NY (0.0)
#define NZ (0.0)

__global__ void t8gpu::init_variable(typename SubgridCompressibleEulerSolver::subgrid_type::Accessor<SubgridCompressibleEulerSolver::float_type> variables) {
  int const e_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;
  int const k = threadIdx.z;

  variables(e_idx, i, j, k) = (e_idx == 1 && i ==0 && j==0 && k==0) ? 1.0 : 0.0;
}

__global__ void t8gpu::compute_inner_fluxes(typename SubgridCompressibleEulerSolver::subgrid_type::Accessor<SubgridCompressibleEulerSolver::float_type> density,
					    typename SubgridCompressibleEulerSolver::subgrid_type::Accessor<SubgridCompressibleEulerSolver::float_type> fluxes,
					    SubgridCompressibleEulerSolver::float_type const* volumes) {
  using SubgridType = typename SubgridCompressibleEulerSolver::subgrid_type;
  using float_type = typename SubgridCompressibleEulerSolver::float_type;

  int const e_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;
  int const k = threadIdx.z;

  float_type v[3] = {NX, NY, NZ};

  float_type volume = volumes[e_idx];
  float_type edge_length = cbrt(volume) / static_cast<float_type>(SubgridType::extent<0>);
  float_type surface = edge_length*edge_length; // TODO

  __shared__ float_type shared_fluxes[SubgridType::template size];

  shared_fluxes[SubgridType::flat_index(i,j,k)] = 0.0;

  if (i < 3) {
    float_type n[3] = {1.0, 0.0, 0.0};

    float_type scalar_product = v[0]*n[0] + v[1]*n[1] + v[2]*n[2];

    float_type flux = surface*scalar_product*(scalar_product > 0 ? density(e_idx, i, j, k) : density(e_idx, i+1, j, k));

    shared_fluxes[SubgridType::flat_index(i,j,k)] = flux;
  }
  __syncthreads();

  if (i < 3)
  fluxes(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i,j,k)];

  if (i > 0)
  fluxes(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i-1,j,k)];

  if (j < 3) {
    float_type n[3] = {0.0, 1.0, 0.0};

    float_type scalar_product = v[0]*n[0] + v[1]*n[1] + v[2]*n[2];

    float_type flux = surface*scalar_product*(scalar_product > 0 ? density(e_idx, i, j, k) : density(e_idx, i, j+1, k));

    shared_fluxes[SubgridType::flat_index(i,j,k)] = flux;
  }
  __syncthreads();

  if (j < 3)
  fluxes(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i,j,k)];

  if (j > 0)
  fluxes(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i,j-1,k)];

  if (k < 3) {
    float_type n[3] = {0.0, 0.0, 1.0};

    float_type scalar_product = v[0]*n[0] + v[1]*n[1] + v[2]*n[2];

    float_type flux = surface*scalar_product*(scalar_product > 0 ? density(e_idx, i, j, k) : density(e_idx, i, j, k+1));

    shared_fluxes[SubgridType::flat_index(i,j,k)] = flux;
  }
  __syncthreads();

  if (k < 3)
  fluxes(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i,j,k)];

  if (k > 0)
  fluxes(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i,j,k-1)];

}

__global__ void t8gpu::compute_outer_fluxes(SubgridMeshConnectivityAccessor<SubgridCompressibleEulerSolver::float_type, SubgridCompressibleEulerSolver::subgrid_type> connectivity,
					    t8_locidx_t const* face_level_difference,
					    t8_locidx_t const* face_neighbor_offset,
					    SubgridMemoryAccessorAll<VariableList, SubgridCompressibleEulerSolver::subgrid_type> density,
					    SubgridMemoryAccessorAll<VariableList, SubgridCompressibleEulerSolver::subgrid_type> fluxes) {
  using float_type = typename SubgridCompressibleEulerSolver::float_type;

  int const f_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;

  t8_locidx_t level_difference = face_level_difference[f_idx];

  int double_stride = (level_difference == 0) ? 2 : 1;

  int offset[3] = {
    face_neighbor_offset[3*f_idx],
    face_neighbor_offset[3*f_idx+1],
    face_neighbor_offset[3*f_idx+2]
  };

  float_type face_surface = connectivity.get_face_surface(f_idx);

  auto [l_idx, r_idx] = connectivity.get_face_neighbor_indices(f_idx);

  int l_rank  = connectivity.get_element_owner_rank(l_idx);
  int l_index = connectivity.get_element_owner_remote_index(l_idx);

  int r_rank  = connectivity.get_element_owner_rank(r_idx);
  int r_index = connectivity.get_element_owner_remote_index(r_idx);

  auto [nx, ny, nz] = connectivity.get_face_normal(f_idx);

  auto rho_l = density.get(l_rank, Rho);
  auto rho_r = density.get(r_rank, Rho);

  auto flux_l = fluxes.get(l_rank, Rho);
  auto flux_r = fluxes.get(r_rank, Rho);

  int anchor_l[3] = {0, 0, 0};

  int anchor_r[3] = {
    offset[0],
    offset[1],
    offset[2]
  };

  int stride_i[3] = {0, 0, 0};
  int stride_j[3] = {0, 0, 0};

  if (nx == 1.0) {
    anchor_l[0] = 3;

    stride_i[1] = 1;
    stride_j[2] = 1;
  }
  if (nx == -1.0) {
    stride_i[1] = 1;
    stride_j[2] = 1;
  }

  if (ny == 1.0) {
    anchor_l[1] = 3;

    stride_i[0] = 1;
    stride_j[2] = 1;
  }

  if (ny == -1.0) {
    stride_i[0] = 1;
    stride_j[2] = 1;
  }

  if (nz == 1.0) {
    anchor_l[2] = 3;

    stride_i[0] = 1;
    stride_j[1] = 1;
  }

  if (nz == -1.0) {
    stride_i[0] = 1;
    stride_j[1] = 1;
  }

  int l_i = anchor_l[0] + i*stride_i[0] + j*stride_j[0];
  int l_j = anchor_l[1] + i*stride_i[1] + j*stride_j[1];
  int l_k = anchor_l[2] + i*stride_i[2] + j*stride_j[2];

  int r_i = anchor_r[0] + double_stride*(i*stride_i[0] + j*stride_j[0])/2;
  int r_j = anchor_r[1] + double_stride*(i*stride_i[1] + j*stride_j[1])/2;
  int r_k = anchor_r[2] + double_stride*(i*stride_i[2] + j*stride_j[2])/2;

  float_type rho_left = rho_l(l_index, l_i, l_j, l_k);

  float_type rho_right = rho_r(r_index, r_i, r_j, r_k);

  float_type v[3] = {NX, NY, NZ};

  float_type scalar_product = v[0]*nx + v[1]*ny + v[2]*nz;

  float_type surface = face_surface / 16.0;

  float_type flux = scalar_product*surface*(scalar_product > 0 ? rho_left : rho_right);

  atomicAdd(&flux_l(l_index, l_i, l_j, l_k), -flux);
  atomicAdd(&flux_r(r_index, r_i, r_j, r_k),  flux);
}

__global__ void t8gpu::euler_update_density(SubgridCompressibleEulerSolver::subgrid_type::Accessor<SubgridCompressibleEulerSolver::float_type> density_prev,
					    SubgridCompressibleEulerSolver::subgrid_type::Accessor<SubgridCompressibleEulerSolver::float_type> density_next,
					    SubgridCompressibleEulerSolver::subgrid_type::Accessor<SubgridCompressibleEulerSolver::float_type> fluxes,
					    SubgridCompressibleEulerSolver::float_type const* volumes,
				       SubgridCompressibleEulerSolver::float_type delta_t) {
  using subgrid_type = typename SubgridCompressibleEulerSolver::subgrid_type;
  using float_type = typename SubgridCompressibleEulerSolver::float_type;

  int const e_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;
  int const k = threadIdx.z;

  float_type volume = volumes[e_idx] / static_cast<float_type>(subgrid_type::template extent<0> * subgrid_type::template extent<1> * subgrid_type::template extent<2>);

  density_next(e_idx, i, j, k) = density_prev(e_idx, i, j, k) - delta_t/volume*fluxes(e_idx, i, j, k);
  fluxes(e_idx, i, j, k) = 0.0;
}

__global__ void t8gpu::compute_refinement_criteria(typename SubgridCompressibleEulerSolver::subgrid_type::Accessor<SubgridCompressibleEulerSolver::float_type> density,
						   SubgridCompressibleEulerSolver::float_type* refinement_criteria,
						   SubgridCompressibleEulerSolver::float_type const* volumes,
						   t8_locidx_t num_local_elements) {
  using subgrid_type = typename SubgridCompressibleEulerSolver::subgrid_type;
  using float_type = typename SubgridCompressibleEulerSolver::float_type;

  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_local_elements)
    return;

  float_type average_density = static_cast<float_type>(0.0);

  for (size_t p=0; p<subgrid_type::template extent<0>; p++) {
    for (size_t q=0; q<subgrid_type::template extent<1>; q++) {
      for (size_t r=0; r<subgrid_type::template extent<2>; r++) {
	average_density += density(i, p, q, r);
      }
    }
  }
  average_density /= static_cast<float_type>(subgrid_type::size);

  refinement_criteria[i] = 10.1 - abs(average_density);
}
