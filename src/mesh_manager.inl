#include <array>
#include <mesh_manager.h>
#include <thrust/host_vector.h>

#include <utils/cuda.h>
#include <utils/profiling.h>
#include <timestepping/ssp_runge_kutta.h>

#include <t8.h>
#include <t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_examples.h>
#include <t8_forest/t8_forest.h>
#include <t8_forest/t8_forest_io.h>
#include <t8_forest/t8_forest_iterate.h>
#include <t8_forest/t8_forest_partition.h>
#include <t8_schemes/t8_default/t8_default.hxx>
#include <t8_element_c_interface.h>

template<typename VariableType, typename StepType, size_t dim>
t8gpu::MeshManager<VariableType, StepType, dim>::MeshManager(sc_MPI_Comm comm,
							     t8_scheme_cxx_t* scheme,
							     t8_cmesh_t cmesh,
							     t8_forest_t forest)
  : m_comm(comm),
    m_scheme(scheme),
    m_cmesh(cmesh),
    m_forest(forest),
    t8gpu::MemoryManager<VariableType, StepType>(t8_forest_get_local_num_elements(forest), comm) {

  MPI_Comm_size(m_comm, &m_nb_ranks);
  MPI_Comm_rank(m_comm, &m_rank);

  m_num_ghost_elements = t8_forest_get_num_ghosts(m_forest);
  m_num_local_elements = t8_forest_get_local_num_elements(m_forest);

  m_ranks.resize(m_num_local_elements + m_num_ghost_elements);
  m_indices.resize(m_num_local_elements + m_num_ghost_elements);
  for (t8_locidx_t i=0; i<m_num_local_elements; i++) {
    m_ranks[i] = m_rank;
    m_indices[i] = i;
  }
  sc_array* sc_array_ranks_wrapper {sc_array_new_data(m_ranks.data(), sizeof(int), m_num_local_elements + m_num_ghost_elements)};
  t8_forest_ghost_exchange_data(m_forest, sc_array_ranks_wrapper);
  sc_array_destroy(sc_array_ranks_wrapper);

  sc_array* sc_array_indices_wrapper {sc_array_new_data(m_indices.data(), sizeof(t8_locidx_t), m_num_local_elements + m_num_ghost_elements)};
  t8_forest_ghost_exchange_data(m_forest, sc_array_indices_wrapper);
  sc_array_destroy(sc_array_indices_wrapper);

  m_device_ranks = m_ranks;
  m_device_indices = m_indices;

  // UserData* forest_user_data = static_cast<UserData*>(malloc(sizeof(UserData)));
  UserData* forest_user_data = new UserData();
  assert(forest_user_data != nullptr);

  forest_user_data->element_refinement_criteria = &m_element_refinement_criteria;
  t8_forest_set_user_data(m_forest, forest_user_data);

  m_element_refinement_criteria.resize(m_num_local_elements);
 }

template<typename VariableType, typename StepType, size_t dim>
t8gpu::MeshManager<VariableType, StepType, dim>::~MeshManager() {
  UserData* forest_user_data = {static_cast<UserData*>(t8_forest_get_user_data(m_forest))};
  delete forest_user_data;

  t8_forest_unref(&m_forest);
  t8_cmesh_destroy(&m_cmesh);
}

template<typename VariableType, typename StepType, size_t dim>
template<typename Func>
void t8gpu::MeshManager<VariableType, StepType, dim>::initialize_variables(Func func) {

  std::array<thrust::host_vector<float_type>, nb_variables> host_variables {};
  for (size_t k=0; k<nb_variables; k++) {
    host_variables[k].resize(m_num_local_elements);
  }

  std::array<float_type*, nb_variables> array {};
  for (size_t k=0; k<nb_variables; k++) {
    array[k] = host_variables[k].data();
  }
  MemoryAccessorOwn<VariableType> host_variable_memory {array};
  thrust::host_vector<float_type> element_volume(m_num_local_elements);

  t8_locidx_t num_local_trees = t8_forest_get_num_local_trees(m_forest);
  t8_locidx_t element_idx = 0;
  for (t8_locidx_t tree_idx = 0; tree_idx < num_local_trees; tree_idx++) {
    t8_eclass_t tree_class {t8_forest_get_tree_class(m_forest, tree_idx)};
    t8_eclass_scheme_c* eclass_scheme {t8_forest_get_eclass_scheme(m_forest, tree_class)};

    t8_locidx_t num_elements_in_tree {t8_forest_get_tree_num_elements(m_forest, tree_idx)};
    for (t8_locidx_t tree_element_idx = 0; tree_element_idx < num_elements_in_tree; tree_element_idx++) {
      const t8_element_t* element {t8_forest_get_element_in_tree(m_forest, tree_idx, tree_element_idx)};

      element_volume[element_idx] = static_cast<float_type>(t8_forest_element_volume(m_forest, tree_idx, element));
      func(host_variable_memory, m_forest, tree_idx, element, element_idx);

      element_idx++;
    }
  }

  // resize shared and owned element variables
  this->resize(m_num_local_elements);

  // fill all variables with zeros
  float_type* device_element_data_ptr {this->get_own_variable(static_cast<StepType>(0), static_cast<VariableType>(0))};
  T8GPU_CUDA_CHECK_ERROR(cudaMemset(device_element_data_ptr, 0, (nb_variables*nb_steps+1)*sizeof(float_type)*m_num_local_elements));

  // m_element_refinement_criteria.resize(m_num_local_elements);
  // m_device_element_refinement_criteria.resize(m_num_local_elements);

  // copy new shared element variables
  for (size_t k=0; k<nb_variables; k++) {
    this->set_variable(static_cast<StepType>(0), static_cast<VariableType>(k), host_variables[k]);
  }

  this->set_volume(std::move(element_volume));

  compute_edge_connectivity();
  m_device_face_neighbors = m_face_neighbors;
  m_device_face_normals = m_face_normals;
  m_device_face_area = m_face_area;
}

template<typename VariableType, typename StepType, size_t dim>
int t8gpu::MeshManager<VariableType, StepType, dim>::adapt_callback_iteration(t8_forest_t forest, t8_forest_t forest_from, t8_locidx_t which_tree, t8_locidx_t lelement_id, t8_eclass_scheme_c* ts,
									 const int is_family, const int num_elements, t8_element_t* elements[]) {
  t8gpu::MeshManager<VariableType, StepType, dim>::UserData* forest_user_data = static_cast<t8gpu::MeshManager<VariableType, StepType, dim>::UserData*>(t8_forest_get_user_data(forest_from));
  assert(forest_user_data != nullptr);

  t8_locidx_t element_level {ts->t8_element_level(elements[0])};

  t8_locidx_t tree_offset = t8_forest_get_tree_element_offset(forest_from, which_tree);

  float_type b = static_cast<float_type>(10.0);

  if (element_level < max_level) {
    float_type criteria = (*forest_user_data->element_refinement_criteria)[tree_offset + lelement_id];

    if (criteria > b) {
      return 1;
    }
  }
  if (element_level > min_level && is_family) {
    float_type criteria = 0.0;
    for (size_t i = 0; i < 4; i++) {
      criteria += (*forest_user_data->element_refinement_criteria)[tree_offset + lelement_id + i] / float_type{4.0};
    }

    if (criteria < b) {
      return -1;
    }
  }

  return 0;
}

template<typename VariableType, typename StepType, size_t dim>
void t8gpu::MeshManager<VariableType, StepType, dim>::refine(const thrust::host_vector<int>& refinement_criteria, StepType step) {
  t8_forest_ref(m_forest);
  assert(t8_forest_is_committed(m_forest));

  assert(refinement_criteria.size() == m_num_local_elements);
  assert(m_element_refinement_criteria.size() == m_num_local_elements);
  m_element_refinement_criteria = refinement_criteria;

  t8_forest_t adapted_forest {};
  t8_forest_init(&adapted_forest);
  t8_forest_set_adapt(adapted_forest, m_forest, adapt_callback_iteration, false);
  t8_forest_set_ghost(adapted_forest, true, T8_GHOST_FACES);
  t8_forest_set_balance(adapted_forest, m_forest, true);
  t8_forest_commit(adapted_forest);

  t8_locidx_t old_idx = 0;
  t8_locidx_t new_idx = 0;

  t8_locidx_t num_new_elements {t8_forest_get_local_num_elements(adapted_forest)};
  t8_locidx_t num_old_elements {t8_forest_get_local_num_elements(m_forest)};

  thrust::host_vector<float_type> adapted_element_variable(num_new_elements);
  thrust::host_vector<float_type> adapted_element_volume(num_new_elements);
  thrust::host_vector<t8_locidx_t> element_adapt_data(num_new_elements + 1);

  thrust::host_vector<t8_locidx_t> old_levels(num_old_elements);
  thrust::host_vector<t8_locidx_t> new_levels(num_new_elements);

  t8_locidx_t num_old_local_trees = {t8_forest_get_num_local_trees(m_forest)};
  t8_locidx_t num_new_local_trees = {t8_forest_get_num_local_trees(m_forest)};

  t8_locidx_t current_idx = 0;
  for (t8_locidx_t tree_idx = 0; tree_idx < num_old_local_trees; tree_idx++) {
    t8_eclass_t old_tree_class {t8_forest_get_tree_class(m_forest, tree_idx)};
    t8_eclass_scheme_c* old_scheme = {t8_forest_get_eclass_scheme(m_forest, old_tree_class)};

    t8_locidx_t num_elements_in_tree {t8_forest_get_tree_num_elements(m_forest, tree_idx)};

    for (t8_locidx_t elem_idx = 0; elem_idx < num_elements_in_tree; elem_idx++) {
      t8_element_t const* element {t8_forest_get_element_in_tree(m_forest, tree_idx, elem_idx)};
      old_levels[current_idx] = old_scheme->t8_element_level(element);
      current_idx++;
    }
  }

  current_idx = 0;
  for (t8_locidx_t tree_idx = 0; tree_idx < num_new_local_trees; tree_idx++) {
    t8_eclass_t new_tree_class {t8_forest_get_tree_class(adapted_forest, tree_idx)};
    t8_eclass_scheme_c* new_scheme = {t8_forest_get_eclass_scheme(adapted_forest, new_tree_class)};

    t8_locidx_t num_elements_in_tree {t8_forest_get_tree_num_elements(adapted_forest, tree_idx)};

    for (t8_locidx_t elem_idx = 0; elem_idx < num_elements_in_tree; elem_idx++) {
      t8_element_t const* element {t8_forest_get_element_in_tree(adapted_forest, tree_idx, elem_idx)};
      new_levels[current_idx] = new_scheme->t8_element_level(element);
      current_idx++;
    }
  }

  while (old_idx < num_old_elements && new_idx < num_new_elements) {
    int old_level = old_levels[old_idx];
    int new_level = new_levels[new_idx];

    if (old_level < new_level) {  // refined
      for (size_t i = 0; i < 4; i++) {
        element_adapt_data[new_idx + i] = old_idx;
      }
      old_idx += 1;
      new_idx += 4;
    } else if (old_level > new_level) {  // coarsened
      for (size_t i = 0; i < 4; i++) {
      }
      element_adapt_data[new_idx] = old_idx;
      old_idx += 4;
      new_idx += 1;
    } else {
      element_adapt_data[new_idx] = old_idx;
      old_idx += 1;
      new_idx += 1;
    }
  }
  element_adapt_data[new_idx] = old_idx;

  t8gpu::MeshManager<VariableType, StepType, dim>::UserData* forest_user_data {static_cast<t8gpu::MeshManager<VariableType, StepType, dim>::UserData*>(t8_forest_get_user_data(m_forest))};
  assert(forest_user_data != nullptr);

  t8_forest_set_user_data(adapted_forest, forest_user_data);
  t8_forest_unref(&m_forest);

  thrust::device_vector<float_type> device_new_conserved_variables(num_new_elements*nb_variables);

  std::array<float_type* __restrict__, nb_variables> new_variables {};
  for (size_t k=0; k<nb_variables; k++) {
    new_variables[k] = thrust::raw_pointer_cast(device_new_conserved_variables.data()) + sizeof(float_type)*num_new_elements;
  }

  thrust::device_vector<float_type> device_element_volume_adapted(num_new_elements);
  t8_locidx_t* device_element_adapt_data {};
  T8GPU_CUDA_CHECK_ERROR(cudaMalloc(&device_element_adapt_data, (num_new_elements + 1) * sizeof(t8_locidx_t)));
  T8GPU_CUDA_CHECK_ERROR(
      cudaMemcpy(device_element_adapt_data, element_adapt_data.data(), element_adapt_data.size() * sizeof(t8_locidx_t), cudaMemcpyHostToDevice));
  const int thread_block_size = 256;
  const int adapt_num_blocks = (num_new_elements + thread_block_size - 1) / thread_block_size;
  adapt_variables_and_volume<VariableType><<<adapt_num_blocks, thread_block_size>>>(
      this->get_own_variables(step),
      new_variables,
      this->get_own_volume(),
      thrust::raw_pointer_cast(device_element_volume_adapted.data()),
      device_element_adapt_data, num_new_elements);
  T8GPU_CUDA_CHECK_LAST_ERROR();
  T8GPU_CUDA_CHECK_ERROR(cudaFree(device_element_adapt_data));

  // resize shared and owned element variables
  this->resize(num_new_elements);

  m_element_refinement_criteria.resize(num_new_elements);
  m_device_refinement_criteria.resize(num_new_elements);

  // // fill fluxes device element variable
  // float_type* device_element_rho_fluxes_ptr {this->get_own_variable(Fluxes, Rho)};
  // T8GPU_CUDA_CHECK_ERROR(cudaMemset(device_element_rho_fluxes_ptr, 0, 5*sizeof(float_type)*num_new_elements)); // cleanup tampered with flux variable

  for (int k=0; k<nb_variables; k++) {
    this->set_variable(step, static_cast<VariableType>(k), new_variables[k]);
  }
  this->set_volume(std::move(device_element_volume_adapted));

  m_forest = adapted_forest;

  m_num_ghost_elements = t8_forest_get_num_ghosts(m_forest);
  m_num_local_elements = t8_forest_get_local_num_elements(m_forest);
}

template<typename VariableType, typename StepType, size_t dim>
__global__ void t8gpu::adapt_variables_and_volume(MemoryAccessorOwn<VariableType> variables_old,
						  std::array<typename variable_traits<VariableType>::float_type* __restrict__, variable_traits<VariableType>::nb_variables> variables_new,
						  typename variable_traits<VariableType>::float_type const* __restrict__ volume_old,
						  typename variable_traits<VariableType>::float_type* __restrict__       volume_new,
						  t8_locidx_t* adapt_data,
						  int nb_new_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_new_elements) return;

  int diff = adapt_data[i + 1] - adapt_data[i];
  int nb_elements_sum = max(1, diff);

  volume_new[i] = volume_old[adapt_data[i]] * ((diff == 0 ? 0.25 : (diff == 1 ? 1.0 : 4.0)));
  if (i > 0 && adapt_data[i - 1] == adapt_data[i]) {
    volume_new[i] = volume_old[adapt_data[i]] * 0.25;
  }

  for (int k=0; k<variable_traits<VariableType>::nb_variables; k++) {
    variables_new[k][i] = 0.0;

    for (int j = 0; j < nb_elements_sum; j++) {
      variables_new[k][i] += variables_old.get(k)[adapt_data[i] + j] / static_cast<typename variable_traits<VariableType>::float_type>(nb_elements_sum);
    }
  }
}

template<typename VariableType, typename StepType, size_t dim>
void t8gpu::MeshManager<VariableType, StepType, dim>::compute_edge_connectivity() {}

template<typename VariableType, typename StepType, size_t dim>
t8gpu::MeshConnectivityAccessor<typename t8gpu::MeshManager<VariableType, StepType, dim>::float_type, dim> t8gpu::MeshManager<VariableType, StepType, dim>::get_connectivity_information() const {
  return {m_ranks, m_indices, m_face_neighbors, m_face_normals, m_face_area};
}
