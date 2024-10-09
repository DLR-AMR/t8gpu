#include <t8gpu/timestepping/ssp_runge_kutta.h>
#include <t8gpu/utils/cuda.h>
#include <t8gpu/utils/meta.h>

#include "kernels.h"

template<typename SubgridType>
__global__ std::enable_if_t<SubgridType::rank == 3, void> initialize_variables(
    t8gpu::SubgridMemoryAccessorOwn<VariableList, SubgridType>              variables,
    typename SubgridCompressibleEulerSolver<SubgridType>::float_type const* centers,
    t8_locidx_t const*                                                      levels) {
  using float_type = typename SubgridCompressibleEulerSolver<SubgridType>::float_type;

  int const e_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;
  int const k = threadIdx.z;

  auto [rho, rho_v1, rho_v2, rho_v3, rho_e] = variables.get(Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);

  t8_locidx_t level = levels[e_idx];

  float_type subgrid_edge_length =
      pow(0.5, static_cast<float_type>(t8gpu::meta::log2_v<SubgridType::template extent<0>>));

  float_type center[3] = {
      static_cast<float_type>(centers[3 * e_idx] - 0.5 * pow(0.5, level) + 0.5 * subgrid_edge_length * pow(0.5, level) +
                              i * subgrid_edge_length * pow(0.5, level)),
      static_cast<float_type>(centers[3 * e_idx + 1] - 0.5 * pow(0.5, level) +
                              0.5 * subgrid_edge_length * pow(0.5, level) + j * subgrid_edge_length * pow(0.5, level)),
      static_cast<float_type>(centers[3 * e_idx + 2] - 0.5 * pow(0.5, level) +
                              0.5 * subgrid_edge_length * pow(0.5, level) + k * subgrid_edge_length * pow(0.5, level))};

  float_type x = float_type{center[0]};
  float_type y = float_type{center[1]};
  float_type z = float_type{center[2]};

  float_type gamma = float_type{1.4};
  float_type sigma = 0.05f / sqrt(2.0f);

  rho(e_idx, i, j, k) = static_cast<float_type>(abs(z - 0.5) < 0.25 ? 2.0 : 1.0);

  rho_v1(e_idx, i, j, k) = static_cast<float_type>(abs(z - 0.5) < 0.25 ? -0.5 : 0.5);
  rho_v2(e_idx, i, j, k) = 0.0;
  rho_v3(e_idx, i, j, k) = static_cast<float_type>(rho(e_idx, i, j, k) *
                                                   (0.1 * sin(4.0f * M_PI * (x - 0.5)) *
                                                    (exp(-((z - 0.75f) / (2 * sigma)) * ((z - 0.75f) / (2 * sigma))) +
                                                     exp(-((z - 0.25f) / (2 * sigma)) * ((z - 0.25f) / (2 * sigma))))));

  rho_e(e_idx, i, j, k) =
      float_type{2.5} / (gamma - float_type{1.0}) +
      float_type{0.5} *
          (rho_v1(e_idx, i, j, k) * rho_v1(e_idx, i, j, k) + rho_v2(e_idx, i, j, k) * rho_v2(e_idx, i, j, k) +
           rho_v3(e_idx, i, j, k) * rho_v3(e_idx, i, j, k)) /
          rho(e_idx, i, j, k);
}

template<typename SubgridType>
__global__ std::enable_if_t<SubgridType::rank == 2, void> initialize_variables(
    t8gpu::SubgridMemoryAccessorOwn<VariableList, SubgridType>              variables,
    typename SubgridCompressibleEulerSolver<SubgridType>::float_type const* centers,
    t8_locidx_t const*                                                      levels) {
  using float_type = typename SubgridCompressibleEulerSolver<SubgridType>::float_type;

  int const e_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;

  auto [rho, rho_v1, rho_v2, rho_v3, rho_e] = variables.get(Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);

  t8_locidx_t level = levels[e_idx];

  float_type subgrid_edge_length =
      pow(0.5, static_cast<float_type>(t8gpu::meta::log2_v<SubgridType::template extent<0>>));

  float_type center[3] = {
      static_cast<float_type>(centers[3 * e_idx] - 0.5 * pow(0.5, level) + 0.5 * subgrid_edge_length * pow(0.5, level) +
                              i * subgrid_edge_length * pow(0.5, level)),
      static_cast<float_type>(centers[3 * e_idx + 1] - 0.5 * pow(0.5, level) +
                              0.5 * subgrid_edge_length * pow(0.5, level) + j * subgrid_edge_length * pow(0.5, level))};

  float_type x = float_type{center[0]};
  float_type y = float_type{center[1]};

  float_type gamma = float_type{1.4};
  float_type sigma = 0.05f / sqrt(2.0f);

  rho(e_idx, i, j) = static_cast<float_type>(abs(y - 0.5) < 0.25 ? 2.0 : 1.0);

  rho_v1(e_idx, i, j) = static_cast<float_type>(abs(y - 0.5) < 0.25 ? -0.5 : 0.5);
  rho_v2(e_idx, i, j) =
      static_cast<float_type>(rho(e_idx, i, j) * (0.1 * sin(4.0f * M_PI * (x - 0.5)) *
                                                  (exp(-((y - 0.75f) / (2 * sigma)) * ((y - 0.75f) / (2 * sigma))) +
                                                   exp(-((y - 0.25f) / (2 * sigma)) * ((y - 0.25f) / (2 * sigma))))));
  rho_v3(e_idx, i, j) = 0.0;

  rho_e(e_idx, i, j) = float_type{2.5} / (gamma - float_type{1.0}) +
                       float_type{0.5} *
                           (rho_v1(e_idx, i, j) * rho_v1(e_idx, i, j) + rho_v2(e_idx, i, j) * rho_v2(e_idx, i, j) +
                            rho_v3(e_idx, i, j) * rho_v3(e_idx, i, j)) /
                           rho(e_idx, i, j);
}

template<typename SubgridType>
SubgridCompressibleEulerSolver<SubgridType>::SubgridCompressibleEulerSolver(sc_MPI_Comm      comm,
                                                                            t8_scheme_cxx_t* scheme,
                                                                            t8_cmesh_t       cmesh,
                                                                            t8_forest_t      forest)
    : m_comm{comm},
      m_mesh_manager{comm, scheme, cmesh, forest},
      m_device_face_speed_estimate(m_mesh_manager.get_num_local_faces() +
                                   m_mesh_manager.get_num_local_boundary_faces()) {
  t8_locidx_t num_local_trees = t8_forest_get_num_local_trees(forest);

  thrust::host_vector<float_type>  centers(m_mesh_manager.get_num_local_elements() * 3);
  thrust::host_vector<t8_locidx_t> levels(m_mesh_manager.get_num_local_elements());

  t8_locidx_t element_idx = 0;
  for (t8_locidx_t tree_idx = 0; tree_idx < num_local_trees; tree_idx++) {
    t8_eclass_t         tree_class{t8_forest_get_tree_class(forest, tree_idx)};
    t8_eclass_scheme_c* eclass_scheme{t8_forest_get_eclass_scheme(forest, tree_class)};

    t8_locidx_t num_elements_in_tree{t8_forest_get_tree_num_elements(forest, tree_idx)};
    for (t8_locidx_t tree_element_idx = 0; tree_element_idx < num_elements_in_tree; tree_element_idx++) {
      t8_element_t const* element{t8_forest_get_element_in_tree(forest, tree_idx, tree_element_idx)};

      double center[3];
      t8_forest_element_centroid(forest, tree_idx, element, center);
      centers[3 * element_idx]     = static_cast<float_type>(center[0]);
      centers[3 * element_idx + 1] = static_cast<float_type>(center[1]);
      centers[3 * element_idx + 2] = static_cast<float_type>(center[2]);

      levels[element_idx] = eclass_scheme->t8_element_level(element);

      element_idx++;
    }
  }

  thrust::device_vector<float_type>  device_centers = centers;
  thrust::device_vector<t8_locidx_t> device_levels  = levels;

  initialize_variables<subgrid_type><<<m_mesh_manager.get_num_local_elements(), subgrid_type::block_size>>>(
      m_mesh_manager.get_own_variables(next),
      thrust::raw_pointer_cast(device_centers.data()),
      thrust::raw_pointer_cast(device_levels.data()));
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
}

template<typename SubgridType>
void SubgridCompressibleEulerSolver<SubgridType>::iterate(float_type delta_t) {
  std::swap(prev, next);

  dim3 dim_grid  = {static_cast<unsigned int>(m_mesh_manager.get_num_local_elements())};
  dim3 dim_block = SubgridType::block_size;

  dim3 dim_grid_face(static_cast<unsigned int>(m_mesh_manager.get_num_local_faces()));
  dim3 dim_block_face = SubgridType::rank == 3 ? dim3{SubgridType::template extent<0>, SubgridType::template extent<1>}
                                               : dim3{SubgridType::template extent<0>};

  int num_boundary_faces = m_mesh_manager.get_num_local_boundary_faces();

  // compute fluxes.
  compute_inner_fluxes<SubgridType><<<dim_grid, dim_block>>>(m_mesh_manager.get_own_variables(prev),
                                                             m_mesh_manager.get_own_variables(Fluxes),
                                                             m_mesh_manager.get_own_volume());
  T8GPU_CUDA_CHECK_LAST_ERROR();

  if (num_boundary_faces > 0) {
    compute_boundary_fluxes<SubgridType>
        <<<num_boundary_faces, dim_block_face>>>(m_mesh_manager.get_connectivity_information(),
                                                 m_mesh_manager.get_own_variables(prev),
                                                 m_mesh_manager.get_own_variables(Fluxes));
  }
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  compute_outer_fluxes<SubgridType><<<dim_grid_face, dim_block_face>>>(m_mesh_manager.get_connectivity_information(),
                                                                       m_mesh_manager.get_all_variables(prev),
                                                                       m_mesh_manager.get_all_variables(Fluxes));
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  // step 1.
  t8gpu::timestepping::subgrid::SSP_3RK_step1<VariableList, SubgridType>
      <<<dim_grid, dim_block>>>(m_mesh_manager.get_own_variables(prev),
                                m_mesh_manager.get_own_variables(Step1),
                                m_mesh_manager.get_own_variables(Fluxes),
                                m_mesh_manager.get_own_volume(),
                                delta_t);
  T8GPU_CUDA_CHECK_LAST_ERROR();

  // compute fluxes.
  compute_inner_fluxes<SubgridType><<<dim_grid, dim_block>>>(m_mesh_manager.get_own_variables(Step1),
                                                             m_mesh_manager.get_own_variables(Fluxes),
                                                             m_mesh_manager.get_own_volume());
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  if (num_boundary_faces > 0) {
    compute_boundary_fluxes<SubgridType>
        <<<num_boundary_faces, dim_block_face>>>(m_mesh_manager.get_connectivity_information(),
                                                 m_mesh_manager.get_own_variables(Step1),
                                                 m_mesh_manager.get_own_variables(Fluxes));
  }
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  compute_outer_fluxes<SubgridType><<<dim_grid_face, dim_block_face>>>(m_mesh_manager.get_connectivity_information(),
                                                                       m_mesh_manager.get_all_variables(Step1),
                                                                       m_mesh_manager.get_all_variables(Fluxes));
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  // step 2.
  t8gpu::timestepping::subgrid::SSP_3RK_step2<VariableList, SubgridType>
      <<<dim_grid, dim_block>>>(m_mesh_manager.get_own_variables(prev),
                                m_mesh_manager.get_own_variables(Step1),
                                m_mesh_manager.get_own_variables(Step2),
                                m_mesh_manager.get_own_variables(Fluxes),
                                m_mesh_manager.get_own_volume(),
                                delta_t);
  T8GPU_CUDA_CHECK_LAST_ERROR();

  // compute fluxes.
  compute_inner_fluxes<SubgridType><<<dim_grid, dim_block>>>(m_mesh_manager.get_own_variables(Step2),
                                                             m_mesh_manager.get_own_variables(Fluxes),
                                                             m_mesh_manager.get_own_volume());
  T8GPU_CUDA_CHECK_LAST_ERROR();

  if (num_boundary_faces > 0) {
    compute_boundary_fluxes<SubgridType>
        <<<num_boundary_faces, dim_block_face>>>(m_mesh_manager.get_connectivity_information(),
                                                 m_mesh_manager.get_own_variables(Step2),
                                                 m_mesh_manager.get_own_variables(Fluxes));
  }
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  compute_outer_fluxes<SubgridType><<<dim_grid_face, dim_block_face>>>(m_mesh_manager.get_connectivity_information(),
                                                                       m_mesh_manager.get_all_variables(Step2),
                                                                       m_mesh_manager.get_all_variables(Fluxes));
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  // step 3.
  t8gpu::timestepping::subgrid::SSP_3RK_step3<VariableList, SubgridType>
      <<<dim_grid, dim_block>>>(m_mesh_manager.get_own_variables(prev),
                                m_mesh_manager.get_own_variables(Step2),
                                m_mesh_manager.get_own_variables(next),
                                m_mesh_manager.get_own_variables(Fluxes),
                                m_mesh_manager.get_own_volume(),
                                delta_t);
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);
}

template<typename SubgridType>
void SubgridCompressibleEulerSolver<SubgridType>::save_density_to_vtk(std::string const& prefix) const {
  m_mesh_manager.save_variable_to_vtk(next, Rho, prefix);
}

template<typename SubgridType>
void SubgridCompressibleEulerSolver<SubgridType>::save_mesh_to_vtk(std::string const& prefix) const {
  m_mesh_manager.save_mesh_to_vtk(prefix);
}

template<typename SubgridType>
SubgridCompressibleEulerSolver<SubgridType>::~SubgridCompressibleEulerSolver() {}

template<typename SubgridType>
typename SubgridCompressibleEulerSolver<SubgridType>::float_type
SubgridCompressibleEulerSolver<SubgridType>::compute_integral() const {
  int                             num_local_elements = m_mesh_manager.get_num_local_elements();
  float_type                      local_integral     = 0.0;
  float_type const*               mem{m_mesh_manager.get_own_variable(next, Rho)};
  thrust::host_vector<float_type> variable(num_local_elements * subgrid_type::size);
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(
      variable.data(), mem, sizeof(float_type) * num_local_elements * subgrid_type::size, cudaMemcpyDeviceToHost));

  thrust::host_vector<float_type> volume(num_local_elements);
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(
      volume.data(), m_mesh_manager.get_own_volume(), sizeof(float_type) * num_local_elements, cudaMemcpyDeviceToHost));

  for (t8_locidx_t i = 0; i < num_local_elements * subgrid_type::size; i++) {
    local_integral += volume[i / subgrid_type::size] / static_cast<float_type>(subgrid_type::size) * variable[i];
  }
  float_type global_integral{};
  if constexpr (std::is_same<float_type, double>::value) {
    MPI_Allreduce(&local_integral, &global_integral, 1, MPI_DOUBLE, MPI_SUM, m_comm);
  } else {
    MPI_Allreduce(&local_integral, &global_integral, 1, MPI_FLOAT, MPI_SUM, m_comm);
  }
  return global_integral;
}

template<typename SubgridType>
typename SubgridCompressibleEulerSolver<SubgridType>::float_type
SubgridCompressibleEulerSolver<SubgridType>::compute_timestep() const {
  // float_type local_speed_estimate = thrust::reduce(m_device_face_speed_estimate.begin(),
  //                                                  m_device_face_speed_estimate.end(),
  //                                                  float_type{0.0},
  //                                                  thrust::maximum<float_type>());
  // float_type global_speed_estimate{};
  // if constexpr (std::is_same<float_type, double>::value) {
  //   MPI_Allreduce(&local_speed_estimate, &global_speed_estimate, 1, MPI_DOUBLE, MPI_MAX, m_comm);
  // } else {
  //   MPI_Allreduce(&local_speed_estimate, &global_speed_estimate, 1, MPI_FLOAT, MPI_MAX, m_comm);
  // }

  // return cfl * static_cast<float_type>(std::pow(static_cast<float_type>(0.5), max_level)) / global_speed_estimate;
  std::cout << "This has not been implemented yet" << std::endl;
  exit(EXIT_FAILURE);
  return 0.0;
}

template<typename SubgridType>
void SubgridCompressibleEulerSolver<SubgridType>::adapt() {
  thrust::device_vector<float_type> device_refinement_criteria(m_mesh_manager.get_num_local_elements());

  constexpr int thread_block_size = 256;
  int const     num_blocks = (m_mesh_manager.get_num_local_elements() + (thread_block_size - 1)) / thread_block_size;
  compute_refinement_criteria<SubgridType>
      <<<num_blocks, thread_block_size>>>(m_mesh_manager.get_own_variable(next, Rho),
                                          thrust::raw_pointer_cast(device_refinement_criteria.data()),
                                          m_mesh_manager.get_own_volume(),
                                          m_mesh_manager.get_num_local_elements());
  T8GPU_CUDA_CHECK_LAST_ERROR();

  thrust::host_vector<float_type> refinement_criteria = device_refinement_criteria;

  m_mesh_manager.adapt(refinement_criteria, next);
  m_mesh_manager.partition(next);
  m_mesh_manager.compute_connectivity_information();
}
