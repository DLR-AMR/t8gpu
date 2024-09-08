#include <cstdio>
#include <string>

#include <t8.h>
#include <t8gpu/mesh/subgrid_mesh_manager.h>

using namespace t8gpu;

enum VariableList {
  Rho,     // density
  nb_variables
};

// defines the number of duplicates per variables that we need
enum StepList {
  Step0,   // used for RK3 timestepping
  Step1,   // used for RK3 timestepping
  Step2,   // used for RK3 timestepping
  Step3,   // used for RK3 timestepping
  Fluxes,  // used to store fluxes
  nb_steps
};

template<typename float_type>
__global__ void init_variable(typename Subgrid<4, 4, 4>::Accessor<float_type> m) {
  int const e_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;
  int const k = threadIdx.z;

  m(e_idx, i, j, k) = (e_idx == 1 && i ==0 && j==0 && k==0) ? 1.0 : 0.0;
}

#define NX (1.0)
#define NY (0.0)
#define NZ (0.0)


template<typename float_type>
__global__ void compute_inner_fluxes(typename Subgrid<4, 4, 4>::Accessor<float_type> density,
				     typename Subgrid<4, 4, 4>::Accessor<float_type> fluxes,
				     float_type const* volumes) {
  using SubgridType = Subgrid<4,4,4>;

  int const e_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;
  int const k = threadIdx.z;

  float_type v[3] = {NX, NY, NZ};

  float_type volume = volumes[e_idx];
  float_type edge_length = cbrt(volume) / static_cast<float_type>(Subgrid<4,4,4>::extent<0>);
  float_type surface = edge_length*edge_length; // TODO

  __shared__ float_type shared_fluxes[64];

  shared_fluxes[SubgridType::flat_index(i,j,k)] = 0.0;

  if (i < 3) {
    float_type n[3] = {1.0, 0.0, 0.0};

    float_type scalar_product = v[0]*n[0] + v[1]*n[1] + v[2]*n[2];

    float_type flux = surface*scalar_product*(scalar_product > 0 ? density(e_idx, i, j, k) : density(e_idx, i+1, j, k));

    shared_fluxes[SubgridType::flat_index(i,j,k)] = flux;
  }
  __syncthreads();

  if (i < 3)
  fluxes(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i,j,k)];

  if (i > 0)
  fluxes(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i-1,j,k)];

  if (j < 3) {
    float_type n[3] = {0.0, 1.0, 0.0};

    float_type scalar_product = v[0]*n[0] + v[1]*n[1] + v[2]*n[2];

    float_type flux = surface*scalar_product*(scalar_product > 0 ? density(e_idx, i, j, k) : density(e_idx, i, j+1, k));

    shared_fluxes[SubgridType::flat_index(i,j,k)] = flux;
  }
  __syncthreads();

  if (j < 3)
  fluxes(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i,j,k)];

  if (j > 0)
  fluxes(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i,j-1,k)];

  if (k < 3) {
    float_type n[3] = {0.0, 0.0, 1.0};

    float_type scalar_product = v[0]*n[0] + v[1]*n[1] + v[2]*n[2];

    float_type flux = surface*scalar_product*(scalar_product > 0 ? density(e_idx, i, j, k) : density(e_idx, i, j, k+1));

    shared_fluxes[SubgridType::flat_index(i,j,k)] = flux;
  }
  __syncthreads();

  if (k < 3)
  fluxes(e_idx, i, j, k) += shared_fluxes[SubgridType::flat_index(i,j,k)];

  if (k > 0)
  fluxes(e_idx, i, j, k) -= shared_fluxes[SubgridType::flat_index(i,j,k-1)];

}

template<typename float_type>
__global__ void compute_outer_fluxes(SubgridMeshConnectivityAccessor<float_type, Subgrid<4,4,4>> connectivity,
				     t8_locidx_t const* face_level_difference,
				     t8_locidx_t const* face_neighbor_offset,
				     SubgridMemoryAccessorAll<VariableList, Subgrid<4,4,4>> density,
				     SubgridMemoryAccessorAll<VariableList, Subgrid<4,4,4>> fluxes) {
  int const f_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;

  t8_locidx_t level_difference = face_level_difference[f_idx];

  if (level_difference > 0) {
    printf("Wrong level difference found\n");
  }

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
  // int anchor_r[3] = {0, 0, 0};
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

  atomicAdd(&flux_l(l_index, l_i, l_j, l_k),  flux);
  atomicAdd(&flux_r(r_index, r_i, r_j, r_k), -flux);
}

template<typename float_type>
__global__ void euler_update_density(typename Subgrid<4, 4, 4>::Accessor<float_type> density_prev,
				     typename Subgrid<4, 4, 4>::Accessor<float_type> density_next,
				     typename Subgrid<4, 4, 4>::Accessor<float_type> fluxes,
				     float_type const* volumes,
				     float_type delta_t) {
  int const e_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;
  int const k = threadIdx.z;

  float_type volume = volumes[e_idx] / static_cast<float_type>(Subgrid<4,4,4>::template extent<0> * Subgrid<4,4,4>::template extent<1> * Subgrid<4,4,4>::template extent<2>);

  density_next(e_idx, i, j, k) = density_prev(e_idx, i, j, k) - delta_t/volume*fluxes(e_idx, i, j, k);
  fluxes(e_idx, i, j, k) = 0.0;
}

int main(int argc, char* argv[]) {
  int mpiret = sc_MPI_Init(&argc, &argv);
  SC_CHECK_MPI(mpiret);

  sc_init(sc_MPI_COMM_WORLD, 1, 1, NULL, SC_LP_ESSENTIAL);
  t8_init(SC_LP_PRODUCTION);

  int rank, nb_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nb_ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // We create a dummy frame so that destructors of solver class are
  // called before MPI finalize.
  {
    using float_type = double;

    float_type delta_t = 0.125 / 2.0;

    StepList prev = Step1;
    StepList next = Step0;

    sc_MPI_Comm comm = MPI_COMM_WORLD;
    t8_scheme_cxx_t* scheme = t8_scheme_new_default_cxx();
    t8_cmesh_t cmesh = t8_cmesh_new_periodic(comm, 3);
    t8_forest_t forest = t8_forest_new_uniform(cmesh, scheme, 1, true, comm);

    SubgridMeshManager<VariableList, StepList, Subgrid<4, 4, 4>> mesh_manager {comm, scheme, cmesh, forest};
    dim3 dimGrid(mesh_manager.get_num_local_elements());
    dim3 dimBlock(4, 4, 4);

    thrust::host_vector<float_type> refinement_criteria(mesh_manager.get_num_local_elements(), 1.0);
    mesh_manager.adapt(refinement_criteria, next);
    mesh_manager.compute_connectivity_information();

    dimGrid = {mesh_manager.get_num_local_elements()};
    dimBlock = {4, 4, 4};

    dim3 dimGridFace(mesh_manager.get_num_local_faces());
    dim3 dimBlockFace(4, 4);

    init_variable<double><<<dimGrid, dimBlock>>>(mesh_manager.get_own_variable(next, Rho));
    T8GPU_CUDA_CHECK_LAST_ERROR();
    cudaDeviceSynchronize();

    for(int it=0; it<25; it++) {
      mesh_manager.save_variable_to_vtk(next, Rho, "density" + std::to_string(it));

      compute_inner_fluxes<double><<<dimGrid, dimBlock>>>(mesh_manager.get_own_variable(next, Rho),
							  mesh_manager.get_own_variable(Fluxes, Rho),
							  mesh_manager.get_own_volume());
      T8GPU_CUDA_CHECK_LAST_ERROR();
      cudaDeviceSynchronize();

      compute_outer_fluxes<double><<<dimGridFace, dimBlockFace>>>(mesh_manager.get_connectivity_information(),
								  thrust::raw_pointer_cast(mesh_manager.m_device_face_level_difference.data()),
								  thrust::raw_pointer_cast(mesh_manager.m_device_face_neighbor_offset.data()),
								  mesh_manager.get_all_variables(next),
								  mesh_manager.get_all_variables(Fluxes));
      T8GPU_CUDA_CHECK_LAST_ERROR();
      cudaDeviceSynchronize();
      MPI_Barrier(comm);

      euler_update_density<double><<<dimGrid, dimBlock>>>(mesh_manager.get_own_variable(next, Rho),
							  mesh_manager.get_own_variable(prev, Rho),
							  mesh_manager.get_own_variable(Fluxes, Rho),
							  mesh_manager.get_own_volume(),
							  delta_t);
      T8GPU_CUDA_CHECK_LAST_ERROR();
      cudaDeviceSynchronize();

      std::swap(prev, next);
    }
  }

  sc_finalize();

  mpiret = sc_MPI_Finalize();
  SC_CHECK_MPI(mpiret);

  return 0;
}
