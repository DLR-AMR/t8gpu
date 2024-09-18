#include <t8gpu/timestepping/ssp_runge_kutta.h>
#include <t8gpu/utils/cuda.h>

#include "kernels.h"
#include "solver.h"

using namespace t8gpu;

CompressibleEulerSolver::CompressibleEulerSolver(sc_MPI_Comm      comm,
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
    using float_type = MemoryAccessorOwn<VariableList>::float_type;

    auto [rho, rho_v1, rho_v2, rho_v3, rho_e] = accessor.get(Rho, Rho_v1, Rho_v2, Rho_v3, Rho_e);

    double center[3];
    t8_forest_element_centroid(forest, tree_idx, element, center);

    float_type sigma = float_type{0.2} / sqrt(float_type{2.0});
    float_type gamma = float_type{1.4};

    float_type x = static_cast<float_type>(center[0]);
    float_type y = static_cast<float_type>(center[1]);
    float_type z = static_cast<float_type>(center[2]);

    float_type r = sqrt(x * x + y * y + z * z);

    // vector normal to the surface of the globe.
    std::array<float_type, 3> e_r = {x / r, y / r, z / r};

    // first tangent vector to surface of the globe. cross_prod(e_r, e_z) / norm(...)
    // This vector is tangent to a latitude of the globe.
    std::array<float_type, 3> e_phi = {e_r[1] / sqrt(e_r[1] * e_r[1] + e_r[0] * e_r[0]),
                                       -e_r[0] / sqrt(e_r[1] * e_r[1] + e_r[0] * e_r[0]),
                                       static_cast<float_type>(0.0)};

    // second tangent vector cross_prod(e_r, u_phi)
    // This vector is tangent to a longitude line.
    std::array<float_type, 3> e_theta = {e_r[1] * e_phi[2] - e_r[2] * e_phi[1],
                                         e_r[2] * e_phi[0] - e_r[0] * e_phi[2],
                                         e_r[0] * e_phi[1] - e_r[1] * e_phi[0]};

    float_type phi = static_cast<float_type>((y >= 0.0) ? acos(x / sqrt(x * x + y * y))
                                                        : 2.0 * M_PI - acos(x / sqrt(x * x + y * y)));

    float_type theta = asin(z / r);
    // We set the initial condition to for a Kelvin-Helmholtz test case.
    float_type v_phi = static_cast<float_type>(r * cos(theta) * (theta < 0 ? -0.5 : 0.5));
    float_type v_theta =
        static_cast<float_type>(0.5 * r * sin(2.0 * phi) * (exp(-(theta / (2 * sigma)) * (theta / (2 * sigma)))));

    rho[e_idx] = static_cast<float_type>(theta < 0.0 ? 2.0 : 1.0);

    rho_v1[e_idx] = rho[e_idx] * (v_phi * e_phi[0] + v_theta * e_theta[0]);
    rho_v2[e_idx] = rho[e_idx] * (v_phi * e_phi[1] + v_theta * e_theta[1]);
    rho_v3[e_idx] = rho[e_idx] * (v_phi * e_phi[2] + v_theta * e_theta[2]);

    rho_e[e_idx] = float_type{2.5} / (gamma - float_type{1.0}) +
                   float_type{0.5} *
                       (rho_v1[e_idx] * rho_v1[e_idx] + rho_v2[e_idx] * rho_v2[e_idx] + rho_v3[e_idx] * rho_v3[e_idx]) /
                       rho[e_idx];
  });
}

void CompressibleEulerSolver::iterate(float_type delta_t) {
  std::swap(next, prev);

  // compute fluxes
  constexpr int thread_block_size = 256;
  int const fluxes_num_blocks = (m_mesh_manager.get_num_local_faces() + (thread_block_size - 1)) / thread_block_size;
  kepes_compute_fluxes<<<fluxes_num_blocks, thread_block_size>>>(
      m_mesh_manager.get_connectivity_information(),
      m_mesh_manager.get_all_variables(prev),
      m_mesh_manager.get_all_variables(Fluxes),
      thrust::raw_pointer_cast(m_device_face_speed_estimate.data()));
  T8GPU_CUDA_CHECK_LAST_ERROR();

  int const boundary_num_blocks =
      (m_mesh_manager.get_num_local_boundary_faces() + (thread_block_size - 1)) / thread_block_size;
  if (m_mesh_manager.get_num_local_boundary_faces() > 0) {
    reflective_boundary_condition<<<boundary_num_blocks, thread_block_size>>>(
        m_mesh_manager.get_connectivity_information(),
        m_mesh_manager.get_own_variables(prev),
        m_mesh_manager.get_own_variables(Fluxes),
        thrust::raw_pointer_cast(m_device_face_speed_estimate.data()));
  }
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  // step 1
  int const SSP_num_blocks = (m_mesh_manager.get_num_local_elements() + thread_block_size - 1) / thread_block_size;
  timestepping::SSP_3RK_step1<VariableList>
      <<<SSP_num_blocks, thread_block_size>>>(m_mesh_manager.get_own_variables(prev),
                                              m_mesh_manager.get_own_variables(Step1),
                                              m_mesh_manager.get_own_variables(Fluxes),
                                              m_mesh_manager.get_own_volume(),
                                              delta_t,
                                              m_mesh_manager.get_num_local_elements());
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  // compute fluxes
  kepes_compute_fluxes<<<fluxes_num_blocks, thread_block_size>>>(
      m_mesh_manager.get_connectivity_information(),
      m_mesh_manager.get_all_variables(Step1),
      m_mesh_manager.get_all_variables(Fluxes),
      thrust::raw_pointer_cast(m_device_face_speed_estimate.data()));
  T8GPU_CUDA_CHECK_LAST_ERROR();

  if (m_mesh_manager.get_num_local_boundary_faces() > 0) {
    reflective_boundary_condition<<<boundary_num_blocks, thread_block_size>>>(
        m_mesh_manager.get_connectivity_information(),
        m_mesh_manager.get_own_variables(Step1),
        m_mesh_manager.get_own_variables(Fluxes),
        thrust::raw_pointer_cast(m_device_face_speed_estimate.data()));
  }
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  // step 2
  timestepping::SSP_3RK_step2<VariableList>
      <<<SSP_num_blocks, thread_block_size>>>(m_mesh_manager.get_own_variables(prev),
                                              m_mesh_manager.get_own_variables(Step1),
                                              m_mesh_manager.get_own_variables(Step2),
                                              m_mesh_manager.get_own_variables(Fluxes),
                                              m_mesh_manager.get_own_volume(),
                                              delta_t,
                                              m_mesh_manager.get_num_local_elements());
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  // compute fluxes
  kepes_compute_fluxes<<<fluxes_num_blocks, thread_block_size>>>(
      m_mesh_manager.get_connectivity_information(),
      m_mesh_manager.get_all_variables(Step2),
      m_mesh_manager.get_all_variables(Fluxes),
      thrust::raw_pointer_cast(m_device_face_speed_estimate.data()));
  T8GPU_CUDA_CHECK_LAST_ERROR();

  if (m_mesh_manager.get_num_local_boundary_faces() > 0) {
    reflective_boundary_condition<<<boundary_num_blocks, thread_block_size>>>(
        m_mesh_manager.get_connectivity_information(),
        m_mesh_manager.get_own_variables(Step2),
        m_mesh_manager.get_own_variables(Fluxes),
        thrust::raw_pointer_cast(m_device_face_speed_estimate.data()));
  }
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  // step 3
  timestepping::SSP_3RK_step3<VariableList>
      <<<SSP_num_blocks, thread_block_size>>>(m_mesh_manager.get_own_variables(prev),
                                              m_mesh_manager.get_own_variables(Step2),
                                              m_mesh_manager.get_own_variables(next),
                                              m_mesh_manager.get_own_variables(Fluxes),
                                              m_mesh_manager.get_own_volume(),
                                              delta_t,
                                              m_mesh_manager.get_num_local_elements());
  T8GPU_CUDA_CHECK_LAST_ERROR();
}

void CompressibleEulerSolver::save_conserved_variables_to_vtk(std::string prefix) const {
  std::array<VariableList, 3> momentum = {Rho_v1, Rho_v2, Rho_v3};

  std::vector<MeshManager<VariableList, StepList, 3>::HostVariableInfo> variables;
  variables.push_back(m_mesh_manager.get_host_scalar_variable(next, Rho, "density"));
  variables.push_back(m_mesh_manager.get_host_scalar_variable(next, Rho_e, "energy"));
  variables.push_back(m_mesh_manager.get_host_vector_variable(next, momentum, "momentum"));

  m_mesh_manager.save_variables_to_vtk(std::move(variables), prefix);
}

CompressibleEulerSolver::~CompressibleEulerSolver() {}

CompressibleEulerSolver::float_type CompressibleEulerSolver::compute_integral() const {
  int                             num_local_elements = m_mesh_manager.get_num_local_elements();
  float_type                      local_integral     = 0.0;
  float_type const*               mem{m_mesh_manager.get_own_variable(next, Rho)};
  thrust::host_vector<float_type> variable(num_local_elements);
  T8GPU_CUDA_CHECK_ERROR(
      cudaMemcpy(variable.data(), mem, sizeof(float_type) * num_local_elements, cudaMemcpyDeviceToHost));
  thrust::host_vector<float_type> volume(num_local_elements);
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(
      volume.data(), m_mesh_manager.get_own_volume(), sizeof(float_type) * num_local_elements, cudaMemcpyDeviceToHost));

  for (t8_locidx_t i = 0; i < num_local_elements; i++) {
    local_integral += volume[i] * variable[i];
  }
  float_type global_integral{};
  if constexpr (std::is_same<float_type, double>::value) {
    MPI_Allreduce(&local_integral, &global_integral, 1, MPI_DOUBLE, MPI_SUM, m_comm);
  } else {
    MPI_Allreduce(&local_integral, &global_integral, 1, MPI_FLOAT, MPI_SUM, m_comm);
  }
  return global_integral;
}

CompressibleEulerSolver::float_type CompressibleEulerSolver::compute_timestep() const {
  float_type local_speed_estimate = thrust::reduce(m_device_face_speed_estimate.begin(),
                                                   m_device_face_speed_estimate.end(),
                                                   float_type{0.0},
                                                   thrust::maximum<float_type>());
  float_type global_speed_estimate{};
  if constexpr (std::is_same<float_type, double>::value) {
    MPI_Allreduce(&local_speed_estimate, &global_speed_estimate, 1, MPI_DOUBLE, MPI_MAX, m_comm);
  } else {
    MPI_Allreduce(&local_speed_estimate, &global_speed_estimate, 1, MPI_FLOAT, MPI_MAX, m_comm);
  }

  return cfl *
         static_cast<float_type>(
             std::pow(static_cast<float_type>(0.5), MeshManager<VariableList, StepList, dim>::max_level)) /
         global_speed_estimate;
}

__global__ static void compute_refinement_criteria(
    typename variable_traits<VariableList>::float_type const* __restrict__ fluxes_rho,
    typename variable_traits<VariableList>::float_type const* __restrict__ volume,
    typename variable_traits<VariableList>::float_type* __restrict__ criteria,
    int num_elements) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= num_elements) return;

  criteria[i] = fluxes_rho[i] / cbrt(volume[i]);
}

void CompressibleEulerSolver::adapt() {
  constexpr int thread_block_size = 256;
  int const gradient_num_blocks = (m_mesh_manager.get_num_local_faces() + (thread_block_size - 1)) / thread_block_size;
  estimate_gradient<<<gradient_num_blocks, thread_block_size>>>(
      m_mesh_manager.get_connectivity_information(),
      m_mesh_manager.get_all_variables(next),
      m_mesh_manager.get_all_variables(Fluxes));  // reuse flux variable here
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(m_comm);

  thrust::device_vector<float_type> device_element_refinement_criteria(m_mesh_manager.get_num_local_elements());

  int const fluxes_num_blocks = (m_mesh_manager.get_num_local_elements() + (thread_block_size - 1)) / thread_block_size;
  compute_refinement_criteria<<<fluxes_num_blocks, thread_block_size>>>(
      m_mesh_manager.get_own_variable(Fluxes, Rho),
      m_mesh_manager.get_own_volume(),
      // thrust::raw_pointer_cast(m_device_element_refinement_criteria.data()),
      thrust::raw_pointer_cast(device_element_refinement_criteria.data()),
      m_mesh_manager.get_num_local_elements());
  T8GPU_CUDA_CHECK_LAST_ERROR();

  // m_element_refinement_criteria = m_device_element_refinement_criteria;
  // m_element_refinement_criteria = device_element_refinement_criteria;
  thrust::host_vector<float_type> element_refinement_criteria = device_element_refinement_criteria;

  T8GPU_CUDA_CHECK_ERROR(cudaMemset(
      m_mesh_manager.get_own_variable(Fluxes, Rho), 0, sizeof(float_type) * m_mesh_manager.get_num_local_elements()));

  // m_mesh_manager.refine(m_element_refinement_criteria, next);
  m_mesh_manager.adapt(element_refinement_criteria, next);
  m_mesh_manager.compute_connectivity_information();
  m_device_face_speed_estimate.resize(m_mesh_manager.get_num_local_faces() +
                                      m_mesh_manager.get_num_local_boundary_faces());
}
