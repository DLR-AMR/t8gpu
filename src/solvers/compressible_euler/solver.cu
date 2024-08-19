#include <solvers/compressible_euler/solver.h>
#include <solvers/compressible_euler/kernels.h>
#include <utils/cuda.h>

using namespace t8gpu;

CompressibleEulerSolver::CompressibleEulerSolver(sc_MPI_Comm comm,
						 t8_scheme_cxx_t* scheme,
						 t8_cmesh_t cmesh,
						 t8_forest_t forest)
  : m_comm {comm},
    m_mesh_manager {comm, scheme, cmesh, forest},
    m_device_face_speed_estimate(m_mesh_manager.get_num_local_faces()) {

  m_mesh_manager.initialize_variables([](MemoryAccessorOwn<VariableList>& accessor,
					 t8_forest_t forest,
					 t8_locidx_t tree_idx,
					 const t8_element_t* element,
					 t8_locidx_t e_idx){
    using float_type = MemoryAccessorOwn<VariableList>::float_type;

    auto [rho, rho_v1, rho_v2, rho_v3, rho_e] = accessor.get(Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);

    double center[3];
    t8_forest_element_centroid(forest, tree_idx, element, center);

    float_type sigma = float_type{0.05}/sqrt(float_type{2.0});
    float_type gamma = float_type{1.4};

    double x = center[0]-0.5;
    double y = center[1]-0.5;

    float_type v1 = static_cast<float_type>(std::abs(y) < 0.25 ? -0.5 : 0.5);
    float_type v2 = static_cast<float_type>(0.1*sin(4.0*M_PI*x)*(exp(-((y-0.25)/(2*sigma))*((y-0.25)/(2*sigma)))+exp(-((y+0.25)/(2*sigma))*((y+0.25)/(2*sigma)))));

    rho[e_idx]    = static_cast<float_type>(std::abs(y) < 0.25 ? 2.0 : 1.0);
    rho_v1[e_idx] = rho[e_idx]*v1;
    rho_v2[e_idx] = rho[e_idx]*v2;
    rho_v3[e_idx] = static_cast<float_type>(0.0);
    rho_e[e_idx]  = float_type{2.0}/(gamma-float_type{1.0}) + float_type{0.5}*(rho_v1[e_idx] * rho_v1[e_idx] + rho_v2[e_idx] * rho_v2[e_idx]) / rho[e_idx];
  });

  m_mesh_manager.compute_connectivity_information();
}

void CompressibleEulerSolver::iterate(float_type delta_t) {
  std::swap(next, prev);

  // compute fluxes
  constexpr int thread_block_size = 256;
  const int fluxes_num_blocks = m_mesh_manager.get_num_local_faces() / thread_block_size;
  kepes_compute_fluxes<<<fluxes_num_blocks, thread_block_size>>>(m_mesh_manager.get_all_variables(prev),
								 m_mesh_manager.get_all_variables(Fluxes),
								 m_mesh_manager.get_connectivity_information(),
								 thrust::raw_pointer_cast(m_device_face_speed_estimate.data()),
								 m_mesh_manager.get_num_local_faces());
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  // step 1
  const int SSP_num_blocks = (m_mesh_manager.get_num_local_elements() + thread_block_size - 1) / thread_block_size;
  timestepping::SSP_3RK_step1<VariableList><<<SSP_num_blocks, thread_block_size>>>(m_mesh_manager.get_own_variables(prev),
										   m_mesh_manager.get_own_variables(Step1),
										   m_mesh_manager.get_own_variables(Fluxes),
										   m_mesh_manager.get_own_volume(),
										   delta_t, m_mesh_manager.get_num_local_elements());
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  // compute fluxes
  kepes_compute_fluxes<<<fluxes_num_blocks, thread_block_size>>>(m_mesh_manager.get_all_variables(Step1),
								 m_mesh_manager.get_all_variables(Fluxes),
								 m_mesh_manager.get_connectivity_information(),
								 thrust::raw_pointer_cast(m_device_face_speed_estimate.data()),
								 m_mesh_manager.get_num_local_faces());
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  // step 2
  timestepping::SSP_3RK_step2<VariableList><<<SSP_num_blocks, thread_block_size>>>(m_mesh_manager.get_own_variables(prev),
										   m_mesh_manager.get_own_variables(Step1),
										   m_mesh_manager.get_own_variables(Step2),
										   m_mesh_manager.get_own_variables(Fluxes),
										   m_mesh_manager.get_own_volume(),
										   delta_t, m_mesh_manager.get_num_local_elements());
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  // compute fluxes
  kepes_compute_fluxes<<<fluxes_num_blocks, thread_block_size>>>(m_mesh_manager.get_all_variables(Step2),
								 m_mesh_manager.get_all_variables(Fluxes),
								 m_mesh_manager.get_connectivity_information(),
								 thrust::raw_pointer_cast(m_device_face_speed_estimate.data()),
								 m_mesh_manager.get_num_local_faces());
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  // step 3
  timestepping::SSP_3RK_step3<VariableList><<<SSP_num_blocks, thread_block_size>>>(m_mesh_manager.get_own_variables(prev),
										   m_mesh_manager.get_own_variables(Step2),
										   m_mesh_manager.get_own_variables(next),
										   m_mesh_manager.get_own_variables(Fluxes),
										   m_mesh_manager.get_own_volume(),
										   delta_t, m_mesh_manager.get_num_local_elements());
  T8GPU_CUDA_CHECK_LAST_ERROR();
}

void CompressibleEulerSolver::save_vtk(std::string prefix) const {
  m_mesh_manager.save_variable_to_vtk(next, Rho, prefix);
}


CompressibleEulerSolver::~CompressibleEulerSolver() {}
