#include <t8gpu/utils/cuda.h>

#include "kernels.h"
#include "solver.h"

using namespace t8gpu;

SubgridCompressibleEulerSolver::SubgridCompressibleEulerSolver(sc_MPI_Comm      comm,
							       t8_scheme_cxx_t* scheme,
							       t8_cmesh_t       cmesh,
							       t8_forest_t      forest)
  : m_comm{comm},
    m_mesh_manager{comm, scheme, cmesh, forest},
    m_device_face_speed_estimate(m_mesh_manager.get_num_local_faces() +
				 m_mesh_manager.get_num_local_boundary_faces()) {

  m_mesh_manager.initialize_variables([](MemoryAccessorOwn<VariableList>& accessor,
                                         t8_forest_t                      forest,
                                         t8_locidx_t                      tree_idx,
                                         t8_element_t const*              element,
                                         t8_locidx_t                      e_idx) {
    using float_type = float_type;

    auto rho = accessor.get(Rho);

    double center[3];
    t8_forest_element_centroid(forest, tree_idx, element, center);

    float_type x = static_cast<float_type>(center[0]);
    float_type y = static_cast<float_type>(center[1]);
    float_type z = static_cast<float_type>(center[2]);

    rho[e_idx] = static_cast<float_type>((x < 0.5) ? 0.0 : 1.0);
  });
}

void SubgridCompressibleEulerSolver::iterate(float_type delta_t) {
  std::swap(prev, next);

  dim3 dimGrid = {static_cast<unsigned int>(m_mesh_manager.get_num_local_elements())};
  dim3 dimBlock = {4, 4, 4};

  dim3 dimGridFace(static_cast<unsigned int>(m_mesh_manager.get_num_local_faces()));
  dim3 dimBlockFace(4, 4);

  compute_inner_fluxes<<<dimGrid, dimBlock>>>(m_mesh_manager.get_own_variable(prev, Rho),
					      m_mesh_manager.get_own_variable(Fluxes, Rho),
					      m_mesh_manager.get_own_volume());
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();

  compute_outer_fluxes<<<dimGridFace, dimBlockFace>>>(m_mesh_manager.get_connectivity_information(),
						      thrust::raw_pointer_cast(m_mesh_manager.m_device_face_level_difference.data()),
						      thrust::raw_pointer_cast(m_mesh_manager.m_device_face_neighbor_offset.data()),
						      m_mesh_manager.get_all_variables(prev),
						      m_mesh_manager.get_all_variables(Fluxes));
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  euler_update_density<<<dimGrid, dimBlock>>>(m_mesh_manager.get_own_variable(prev, Rho),
					      m_mesh_manager.get_own_variable(next, Rho),
					      m_mesh_manager.get_own_variable(Fluxes, Rho),
					      m_mesh_manager.get_own_volume(),
					      delta_t);
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
}

void SubgridCompressibleEulerSolver::save_conserved_variables_to_vtk(std::string prefix) const {
  m_mesh_manager.save_variable_to_vtk(next, Rho, prefix);
  // std::array<VariableList, 3> momentum = {Rho_v1, Rho_v2, Rho_v3};

  // std::vector<MeshManager<VariableList, StepList, 3>::HostVariableInfo> variables;
  // variables.push_back(m_mesh_manager.get_host_scalar_variable(next, Rho, "density"));
  // variables.push_back(m_mesh_manager.get_host_scalar_variable(next, Rho_e, "energy"));
  // variables.push_back(m_mesh_manager.get_host_vector_variable(next, momentum, "momentum"));

  // m_mesh_manager.save_variables_to_vtk(std::move(variables), prefix);
}

SubgridCompressibleEulerSolver::~SubgridCompressibleEulerSolver() {}

SubgridCompressibleEulerSolver::float_type SubgridCompressibleEulerSolver::compute_integral() const {
  int                             num_local_elements = m_mesh_manager.get_num_local_elements();
  float_type                      local_integral     = 0.0;
  float_type const*               mem{m_mesh_manager.get_own_variable(next, Rho)};
  thrust::host_vector<float_type> variable(num_local_elements*subgrid_type::size);
  T8GPU_CUDA_CHECK_ERROR(
			 cudaMemcpy(variable.data(), mem, sizeof(float_type) * num_local_elements * subgrid_type::size, cudaMemcpyDeviceToHost));

  thrust::host_vector<float_type> volume(num_local_elements);
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(
				    volume.data(), m_mesh_manager.get_own_volume(), sizeof(float_type) * num_local_elements, cudaMemcpyDeviceToHost));

  for (t8_locidx_t i = 0; i < num_local_elements*subgrid_type::size; i++) {
    local_integral += volume[i / subgrid_type::size] * variable[i];
  }
  float_type global_integral{};
  if constexpr (std::is_same<float_type, double>::value) {
    MPI_Allreduce(&local_integral, &global_integral, 1, MPI_DOUBLE, MPI_SUM, m_comm);
  } else {
    MPI_Allreduce(&local_integral, &global_integral, 1, MPI_FLOAT, MPI_SUM, m_comm);
  }
  return global_integral;
}

SubgridCompressibleEulerSolver::float_type SubgridCompressibleEulerSolver::compute_timestep() const {
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

void SubgridCompressibleEulerSolver::adapt() {
  thrust::host_vector<float_type> refinement_criteria(m_mesh_manager.get_num_local_elements(), 1.0);
  m_mesh_manager.adapt(refinement_criteria, next);
  m_mesh_manager.compute_connectivity_information();
}
