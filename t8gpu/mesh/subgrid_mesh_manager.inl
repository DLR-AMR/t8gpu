#include "subgrid_mesh_manager.h"

#include <thrust/host_vector.h>
#include <array>
#include <type_traits>

#include <t8gpu/utils/cuda.h>
#include <t8gpu/utils/profiling.h>

#include <t8.h>
#include <t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_examples.h>
#include <t8_element_c_interface.h>
#include <t8_forest/t8_forest.h>
#include <t8_forest/t8_forest_io.h>
#include <t8_forest/t8_forest_iterate.h>
#include <t8_forest/t8_forest_partition.h>
#include <t8_schemes/t8_default/t8_default.hxx>

template<typename VariableType, typename StepType, typename SubgridType>
t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::SubgridMeshManager(sc_MPI_Comm      comm,
										   t8_scheme_cxx_t* scheme,
										   t8_cmesh_t       cmesh,
										   t8_forest_t      forest)
    : m_comm{comm},
      m_scheme{scheme},
      m_cmesh{cmesh},
      m_forest{forest},
      t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>{static_cast<size_t>(t8_forest_get_local_num_elements(forest)),
    comm} {
  MPI_Comm_size(m_comm, &m_nb_ranks);
  MPI_Comm_rank(m_comm, &m_rank);

  m_num_ghost_elements = t8_forest_get_num_ghosts(m_forest);
  m_num_local_elements = t8_forest_get_local_num_elements(m_forest);

  m_ranks.resize(m_num_local_elements + m_num_ghost_elements);
  m_indices.resize(m_num_local_elements + m_num_ghost_elements);
  for (t8_locidx_t i = 0; i < m_num_local_elements; i++) {
    m_ranks[i]   = m_rank;
    m_indices[i] = i;
  }
  sc_array* sc_array_ranks_wrapper{
      sc_array_new_data(m_ranks.data(), sizeof(int), m_num_local_elements + m_num_ghost_elements)};
  t8_forest_ghost_exchange_data(m_forest, sc_array_ranks_wrapper);
  sc_array_destroy(sc_array_ranks_wrapper);

  sc_array* sc_array_indices_wrapper{
      sc_array_new_data(m_indices.data(), sizeof(t8_locidx_t), m_num_local_elements + m_num_ghost_elements)};
  t8_forest_ghost_exchange_data(m_forest, sc_array_indices_wrapper);
  sc_array_destroy(sc_array_indices_wrapper);

  m_device_ranks   = m_ranks;
  m_device_indices = m_indices;

  UserData* forest_user_data = new UserData();
  assert(forest_user_data != nullptr);

  forest_user_data->element_refinement_criteria = &m_element_refinement_criteria;
  t8_forest_set_user_data(m_forest, forest_user_data);

  m_element_refinement_criteria.resize(m_num_local_elements);

  this->compute_connectivity_information();
}

template<typename VariableType, typename StepType, typename SubgridType>
t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::~SubgridMeshManager() {
  UserData* forest_user_data = {static_cast<UserData*>(t8_forest_get_user_data(m_forest))};
  delete forest_user_data;

  t8_forest_unref(&m_forest);
  t8_cmesh_destroy(&m_cmesh);
}

template<typename VariableType, typename StepType, typename SubgridType>
int t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::adapt_callback_iteration(t8_forest_t         forest,
											     t8_forest_t         forest_from,
											     t8_locidx_t         which_tree,
											     t8_locidx_t         lelement_id,
											     t8_eclass_scheme_c* ts,
											     int const           is_family,
											     int const           num_elements,
											     t8_element_t*       elements[]) {
  // t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::UserData* forest_user_data =
  //     static_cast<t8gpu::MeshManager<VariableType, StepType, SubgridType>::UserData*>(t8_forest_get_user_data(forest_from));
  // assert(forest_user_data != nullptr);

  // t8_locidx_t element_level{ts->t8_element_level(elements[0])};

  // t8_locidx_t tree_offset = t8_forest_get_tree_element_offset(forest_from, which_tree);

  // float_type b = static_cast<float_type>(10.0);

  // if (element_level < max_level) {
  //   float_type criteria = (*forest_user_data->element_refinement_criteria)[tree_offset + lelement_id];

  //   if (criteria > b) {
  //     return 1;
  //   }
  // }
  // if (element_level > min_level && is_family) {
  //   float_type criteria = 0.0;
  //   for (size_t i = 0; i < 4; i++) {
  //     criteria += (*forest_user_data->element_refinement_criteria)[tree_offset + lelement_id + i] / float_type{4.0};
  //   }

  //   if (criteria < b) {
  //     return -1;
  //   }
  // }

  return 0;
}

template<typename VariableType>
__global__ void adapt_volume(
    typename t8gpu::variable_traits<VariableType>::float_type const* __restrict__ volume_old,
    typename t8gpu::variable_traits<VariableType>::float_type* __restrict__ volume_new,
    t8_locidx_t* adapt_data,
    int          nb_new_elements) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_new_elements) return;

  int diff            = adapt_data[i + 1] - adapt_data[i];
  int nb_elements_sum = max(1, diff);

  volume_new[i] = volume_old[adapt_data[i]] * ((diff == 0 ? 0.125 : (diff == 1 ? 1.0 : 8.0)));
  if (i > 0 && adapt_data[i - 1] == adapt_data[i]) {
    volume_new[i] = volume_old[adapt_data[i]] * 0.125;
  }
}

template<typename VariableType, typename SubgridType>
__global__ void adapt_variables(t8gpu::SubgridMemoryAccessorOwn<VariableType, SubgridType> variables_old,
				std::array<typename t8gpu::variable_traits<VariableType>::float_type* __restrict__,
				t8gpu::variable_traits<VariableType>::nb_variables> variables_new,
				t8_locidx_t* adapt_data) {
  int const e_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;
  int const k = threadIdx.z;

  // We can afford an if statement here as every thread in a thread block takes the same path.
  if ((adapt_data[e_idx + 1] == adapt_data[e_idx]) || (e_idx > 0 && (adapt_data[e_idx] == adapt_data[e_idx - 1]))) { // refinement

    int refinement_index = 0; // the index of the refined element in {0, ..., 7} in z-order.
    while (e_idx - refinement_index >= 0 && adapt_data[e_idx - refinement_index] == adapt_data[e_idx])
      refinement_index++;

    int I = ((refinement_index-1) & 0b1);
    int J = ((refinement_index-1) & 0b10) >> 1;
    int K = ((refinement_index-1) & 0b100) >> 2;

    for (int l=0; l<t8gpu::variable_traits<VariableType>::nb_variables; l++) {
      variables_new[l][e_idx * SubgridType::size + SubgridType::flat_index(i, j, k)] =
	variables_old.get(l)(adapt_data[e_idx],
			     I*blockDim.x/2 + i/2,
			     J*blockDim.y/2 + j/2,
			     K*blockDim.z/2 + k/2);
    }

  } else if ((adapt_data[e_idx + 1] - adapt_data[e_idx]) > 1) { // coarsening
    int I = (i & 0b10) >> 1;
    int J = (i & 0b10) >> 1;
    int K = (i & 0b10) >> 1;

    int z_index = I | (J << 1) | (K << 2);

    for (int l=0; l<t8gpu::variable_traits<VariableType>::nb_variables; l++) {
      variables_new[l][e_idx * SubgridType::size + SubgridType::flat_index(i, j, k)] = 0.0;

      for (int ii = 0; ii<2; ii++) {
	for (int jj = 0; jj<2; jj++) {
	  for (int kk = 0; kk<2; kk++) {
	    variables_new[l][e_idx * SubgridType::size + SubgridType::flat_index(i, j, k)] +=
	      variables_old.get(l)(adapt_data[e_idx] + z_index, 2*(i & 0b1) + ii, 2*(j & 0b1) + jj, 2*(k & 0b1) + kk);
	  }
	}
      }
      variables_new[l][e_idx * SubgridType::size + SubgridType::flat_index(i, j, k)] /= 8.0;
    }
  } else { // no refinement, no coarsening
    for (int l=0; l<t8gpu::variable_traits<VariableType>::nb_variables; l++) {
      variables_new[l][e_idx * SubgridType::size + SubgridType::flat_index(i, j, k)] = variables_old.get(l)(adapt_data[e_idx], i, j, k);
    }
  }
}

template<typename VariableType, typename StepType, typename SubgridType>
void t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::adapt(
    thrust::host_vector<typename t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::float_type> const&
                                                                              refinement_criteria,
    typename t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::step_index_type step) {
  t8_forest_ref(m_forest);
  assert(t8_forest_is_committed(m_forest));

  assert(refinement_criteria.size() == m_num_local_elements);
  assert(m_element_refinement_criteria.size() == m_num_local_elements);
  m_element_refinement_criteria = refinement_criteria;

  t8_forest_t adapted_forest{};
  t8_forest_init(&adapted_forest);
  t8_forest_set_adapt(adapted_forest, m_forest, adapt_callback_iteration, false);
  t8_forest_set_ghost(adapted_forest, true, T8_GHOST_FACES);
  t8_forest_set_balance(adapted_forest, m_forest, true);
  t8_forest_commit(adapted_forest);

  t8_locidx_t old_idx = 0;
  t8_locidx_t new_idx = 0;

  t8_locidx_t num_new_elements{t8_forest_get_local_num_elements(adapted_forest)};
  t8_locidx_t num_old_elements{t8_forest_get_local_num_elements(m_forest)};

  thrust::host_vector<float_type>  adapted_element_variable(num_new_elements);
  thrust::host_vector<float_type>  adapted_element_volume(num_new_elements);
  thrust::host_vector<t8_locidx_t> element_adapt_data(num_new_elements + 1);

  thrust::host_vector<t8_locidx_t> old_levels(num_old_elements);
  thrust::host_vector<t8_locidx_t> new_levels(num_new_elements);

  t8_locidx_t num_old_local_trees = {t8_forest_get_num_local_trees(m_forest)};
  t8_locidx_t num_new_local_trees = {t8_forest_get_num_local_trees(m_forest)};

  t8_locidx_t current_idx = 0;
  for (t8_locidx_t tree_idx = 0; tree_idx < num_old_local_trees; tree_idx++) {
    t8_eclass_t         old_tree_class{t8_forest_get_tree_class(m_forest, tree_idx)};
    t8_eclass_scheme_c* old_scheme = {t8_forest_get_eclass_scheme(m_forest, old_tree_class)};

    t8_locidx_t num_elements_in_tree{t8_forest_get_tree_num_elements(m_forest, tree_idx)};

    for (t8_locidx_t elem_idx = 0; elem_idx < num_elements_in_tree; elem_idx++) {
      t8_element_t const* element{t8_forest_get_element_in_tree(m_forest, tree_idx, elem_idx)};
      old_levels[current_idx] = old_scheme->t8_element_level(element);
      current_idx++;
    }
  }

  current_idx = 0;
  for (t8_locidx_t tree_idx = 0; tree_idx < num_new_local_trees; tree_idx++) {
    t8_eclass_t         new_tree_class{t8_forest_get_tree_class(adapted_forest, tree_idx)};
    t8_eclass_scheme_c* new_scheme = {t8_forest_get_eclass_scheme(adapted_forest, new_tree_class)};

    t8_locidx_t num_elements_in_tree{t8_forest_get_tree_num_elements(adapted_forest, tree_idx)};

    for (t8_locidx_t elem_idx = 0; elem_idx < num_elements_in_tree; elem_idx++) {
      t8_element_t const* element{t8_forest_get_element_in_tree(adapted_forest, tree_idx, elem_idx)};
      new_levels[current_idx] = new_scheme->t8_element_level(element);
      current_idx++;
    }
  }

  while (old_idx < num_old_elements && new_idx < num_new_elements) {
    int old_level = old_levels[old_idx];
    int new_level = new_levels[new_idx];

    constexpr int nb_subelements = (dim == 2) ? 4 : 8;
    if (old_level < new_level) {  // refined
      for (size_t i = 0; i < nb_subelements; i++) {
        element_adapt_data[new_idx + i] = old_idx;
      }
      old_idx += 1;
      new_idx += nb_subelements;
    } else if (old_level > new_level) {  // coarsened
      for (size_t i = 0; i < nb_subelements; i++) {
      }
      element_adapt_data[new_idx] = old_idx;
      old_idx += nb_subelements;
      new_idx += 1;
    } else {
      element_adapt_data[new_idx] = old_idx;
      old_idx += 1;
      new_idx += 1;
    }
  }
  element_adapt_data[new_idx] = old_idx;

  t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::UserData* forest_user_data{
      static_cast<t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::UserData*>(t8_forest_get_user_data(m_forest))};
  assert(forest_user_data != nullptr);

  t8_forest_set_user_data(adapted_forest, forest_user_data);
  t8_forest_unref(&m_forest);

  thrust::device_vector<float_type> device_new_conserved_variables(num_new_elements * nb_variables * SubgridType::size);

  std::array<float_type* __restrict__, nb_variables> new_variables{};
  for (size_t k = 0; k < nb_variables; k++) {
    new_variables[k] = thrust::raw_pointer_cast(device_new_conserved_variables.data()) + k * num_new_elements * SubgridType::size;
  }

  thrust::device_vector<float_type> device_element_volume_adapted(num_new_elements);
  t8_locidx_t*                      device_element_adapt_data{};
  T8GPU_CUDA_CHECK_ERROR(cudaMalloc(&device_element_adapt_data, (num_new_elements + 1) * sizeof(t8_locidx_t)));
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(device_element_adapt_data,
                                    element_adapt_data.data(),
                                    element_adapt_data.size() * sizeof(t8_locidx_t),
                                    cudaMemcpyHostToDevice));

  int const thread_block_size = 256;
  int const adapt_num_blocks  = (num_new_elements + thread_block_size - 1) / thread_block_size;
  adapt_volume<VariableType>
      <<<adapt_num_blocks, thread_block_size>>>(this->get_own_volume(),
                                                thrust::raw_pointer_cast(device_element_volume_adapted.data()),
                                                device_element_adapt_data,
                                                num_new_elements);
  T8GPU_CUDA_CHECK_LAST_ERROR();
  T8GPU_CUDA_CHECK_ERROR(cudaFree(device_element_adapt_data));

  /////// ADAPT VARIABLES
  dim3 dimGrid(num_new_elements);
  dim3 dimBlock(4, 4, 4);
  adapt_variables<VariableType>
    <<<dimGrid, dimBlock>>>(this->get_own_variables(step),
			    new_variables,
			    device_element_adapt_data);
  T8GPU_CUDA_CHECK_LAST_ERROR();
  ///////

  // resize shared and owned element variables
  this->resize(num_new_elements);

  m_element_refinement_criteria.resize(num_new_elements);

  for (int k = 0; k < nb_variables; k++) {
    this->set_variable(step, static_cast<VariableType>(k), new_variables[k]);
  }
  this->set_volume(std::move(device_element_volume_adapted));

  m_forest = adapted_forest;

  m_num_ghost_elements = t8_forest_get_num_ghosts(m_forest);
  m_num_local_elements = t8_forest_get_local_num_elements(m_forest);
}

template<typename VariableType, typename StepType, typename SubgridType>
void t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::compute_connectivity_information() {
  m_ranks.resize(m_num_local_elements + m_num_ghost_elements);
  m_indices.resize(m_num_local_elements + m_num_ghost_elements);
  for (t8_locidx_t i = 0; i < m_num_local_elements; i++) {
    m_ranks[i]   = m_rank;
    m_indices[i] = i;
  }
  sc_array* sc_array_ranks_wrapper{
      sc_array_new_data(m_ranks.data(), sizeof(int), m_num_local_elements + m_num_ghost_elements)};
  t8_forest_ghost_exchange_data(m_forest, sc_array_ranks_wrapper);
  sc_array_destroy(sc_array_ranks_wrapper);

  sc_array* sc_array_indices_wrapper{
      sc_array_new_data(m_indices.data(), sizeof(t8_locidx_t), m_num_local_elements + m_num_ghost_elements)};
  t8_forest_ghost_exchange_data(m_forest, sc_array_indices_wrapper);
  sc_array_destroy(sc_array_indices_wrapper);

  m_device_ranks   = m_ranks;
  m_device_indices = m_indices;

  /** we store the two face neighbors local element per locally owned faces */
  thrust::host_vector<t8_locidx_t> face_neighbors{};

  thrust::host_vector<t8_locidx_t> face_anchors{};
  thrust::host_vector<t8_locidx_t> face_strides{};

  /** We store the element id of the neighbor element to boundary faces */
  thrust::host_vector<t8_locidx_t> boundary_face_neighbors{};

  /** We store face normals interleaved */
  thrust::host_vector<float_type> face_normals{};
  thrust::host_vector<float_type> boundary_face_normals{};

  /** We store the face surface area */
  thrust::host_vector<float_type> face_area{};
  thrust::host_vector<float_type> boundary_face_area{};

  assert(t8_forest_is_committed(m_forest));

  t8_locidx_t num_local_trees{t8_forest_get_num_local_trees(m_forest)};
  t8_locidx_t element_idx = 0;
  for (t8_locidx_t tree_idx = 0; tree_idx < num_local_trees; tree_idx++) {
    t8_eclass_t         tree_class = t8_forest_get_tree_class(m_forest, tree_idx);
    t8_eclass_scheme_c* eclass_scheme{t8_forest_get_eclass_scheme(m_forest, tree_class)};

    t8_locidx_t num_elements_in_tree{t8_forest_get_tree_num_elements(m_forest, tree_idx)};
    for (t8_locidx_t tree_element_idx = 0; tree_element_idx < num_elements_in_tree; tree_element_idx++) {
      t8_element_t const* element{t8_forest_get_element_in_tree(m_forest, tree_idx, tree_element_idx)};

      t8_locidx_t num_faces{eclass_scheme->t8_element_num_faces(element)};
      for (t8_locidx_t face_idx = 0; face_idx < num_faces; face_idx++) {
        int                 num_neighbors{};
        int*                dual_faces{};
        t8_locidx_t*        neighbor_ids{};
        t8_element_t**      neighbors{};
        t8_eclass_scheme_c* neigh_scheme{};
        t8_forest_leaf_face_neighbors(m_forest,
                                      tree_idx,
                                      element,
                                      &neighbors,
                                      face_idx,
                                      &dual_faces,
                                      &num_neighbors,
                                      &neighbor_ids,
                                      &neigh_scheme,
                                      true);

        for (int i = 0; i < num_neighbors; i++) { // we treat the case when the neighboring element is a ghost element.
          if (neighbor_ids[i] >= m_num_local_elements && m_rank < m_ranks[neighbor_ids[i]]) {
            face_neighbors.push_back(element_idx);
            face_neighbors.push_back(neighbor_ids[i]);
            double face_normal[dim];
            t8_forest_element_face_normal(m_forest, tree_idx, element, face_idx, face_normal);
            for (size_t k = 0; k < dim; k++) {
              face_normals.push_back(static_cast<float_type>(face_normal[k]));
            }
            face_area.push_back(
                static_cast<float_type>(t8_forest_element_face_area(m_forest, tree_idx, element, face_idx)) /
                static_cast<float_type>(num_neighbors));
          }
        }

	// we treat the case when the neighboring element is owned by the same rank.
        if ((num_neighbors == 1) && (neighbor_ids[0] < m_num_local_elements) &&
            ((neighbor_ids[0] > element_idx) ||
             (neighbor_ids[0] < element_idx &&
              neigh_scheme[0].t8_element_level(neighbors[0]) < eclass_scheme->t8_element_level(element)))) {
          face_neighbors.push_back(element_idx);
          face_neighbors.push_back(neighbor_ids[0]);
          double face_normal[dim];
          t8_forest_element_face_normal(m_forest, tree_idx, element, face_idx, face_normal);
          for (size_t k = 0; k < dim; k++) {
            face_normals.push_back(static_cast<float_type>(face_normal[k]));
          }
          face_area.push_back(
              static_cast<float_type>(t8_forest_element_face_area(m_forest, tree_idx, element, face_idx)));
        }
        neigh_scheme->t8_element_destroy(num_neighbors, neighbors);
        T8_FREE(neighbors);

        T8_FREE(dual_faces);
        T8_FREE(neighbor_ids);

        if (num_neighbors == 0) {
          boundary_face_neighbors.push_back(element_idx);
          double face_normal[dim];
          t8_forest_element_face_normal(m_forest, tree_idx, element, face_idx, face_normal);
          for (size_t k = 0; k < dim; k++) {
            boundary_face_normals.push_back(static_cast<float_type>(face_normal[k]));
          }
          boundary_face_area.push_back(
              static_cast<float_type>(t8_forest_element_face_area(m_forest, tree_idx, element, face_idx)));
        }
      }

      element_idx++;
    }
  }
  m_num_local_faces          = static_cast<t8_locidx_t>(face_area.size());
  m_num_local_boundary_faces = static_cast<t8_locidx_t>(boundary_face_area.size());

  // we concatenate the inner and boundary face normals.
  m_device_face_neighbors.resize(face_neighbors.size() + boundary_face_neighbors.size());
  cudaMemcpy(thrust::raw_pointer_cast(m_device_face_neighbors.data()),
             thrust::raw_pointer_cast(face_neighbors.data()),
             face_neighbors.size() * sizeof(t8_locidx_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(thrust::raw_pointer_cast(m_device_face_neighbors.data()) + face_neighbors.size(),
             thrust::raw_pointer_cast(boundary_face_neighbors.data()),
             boundary_face_neighbors.size() * sizeof(t8_locidx_t),
             cudaMemcpyHostToDevice);

  // we concatenate the inner and boundary face normals.
  m_device_face_normals.resize(face_normals.size() + boundary_face_normals.size());
  cudaMemcpy(thrust::raw_pointer_cast(m_device_face_normals.data()),
             thrust::raw_pointer_cast(face_normals.data()),
             face_normals.size() * sizeof(float_type),
             cudaMemcpyHostToDevice);
  cudaMemcpy(thrust::raw_pointer_cast(m_device_face_normals.data()) + face_normals.size(),
             thrust::raw_pointer_cast(boundary_face_normals.data()),
             boundary_face_normals.size() * sizeof(float_type),
             cudaMemcpyHostToDevice);

  // we concatenate the inner and boundary face area.
  m_device_face_area.resize(face_area.size() + boundary_face_area.size());
  cudaMemcpy(thrust::raw_pointer_cast(m_device_face_area.data()),
             thrust::raw_pointer_cast(face_area.data()),
             face_area.size() * sizeof(float_type),
             cudaMemcpyHostToDevice);
  cudaMemcpy(thrust::raw_pointer_cast(m_device_face_area.data()) + face_area.size(),
             thrust::raw_pointer_cast(boundary_face_area.data()),
             boundary_face_area.size() * sizeof(float_type),
             cudaMemcpyHostToDevice);
}

template<typename VariableType, typename StepType, typename SubgridType>
t8gpu::SubgridMeshConnectivityAccessor<typename t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::float_type, SubgridType>
t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::get_connectivity_information() const {
  return {thrust::raw_pointer_cast(m_device_ranks.data()),
          thrust::raw_pointer_cast(m_device_indices.data()),
          thrust::raw_pointer_cast(m_device_face_neighbors.data()),
          thrust::raw_pointer_cast(m_device_face_normals.data()),
          thrust::raw_pointer_cast(m_device_face_area.data()),
          m_num_local_faces,
          m_num_local_boundary_faces};
}

template<typename VariableType, typename StepType, typename SubgridType>
int t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::get_num_local_elements() const {
  return m_num_local_elements;
}

template<typename VariableType, typename StepType, typename SubgridType>
int t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::get_num_ghost_elements() const {
  return m_num_ghost_elements;
}

template<typename VariableType, typename StepType, typename SubgridType>
int t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::get_num_local_faces() const {
  return m_num_local_faces;
}

template<typename VariableType, typename StepType, typename SubgridType>
int t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::get_num_local_boundary_faces() const {
  return m_num_local_boundary_faces;
}

template<typename float_type>
__global__ void column_major_to_z_order(float_type const* from,
					float_type* to) {
  int const e_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;
  int const k = threadIdx.z;

  int morton_index = (i & 0b1) | ((j & 0b1) << 1) | ((k & 0b1) << 2)
                   | ((i & 0b10) << 2) | ((j & 0b10) << 3) | ((k & 0b10) << 4);

  to[e_idx*64 + morton_index] = from[e_idx*64 + i + 4*j + 16*k];
}

template<typename VariableType, typename StepType, typename SubgridType>
void t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::save_variable_to_vtk(
    t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::step_index_type     step,
    t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::variable_index_type variable,
    std::string const&                                                   prefix) const {
  static_assert(std::is_same_v<SubgridType, Subgrid<4,4,4>>);

  thrust::device_vector<float_type> device_element_variable(m_num_local_elements * SubgridType::size);

  thrust::host_vector<float_type> element_variable {};

  dim3 dimGrid(m_num_local_elements);
  dim3 dimBlock(4, 4, 4);
  column_major_to_z_order<float_type><<<dimGrid, dimBlock>>>(static_cast<float_type const*>(this->get_own_variable(step, variable)),
							     thrust::raw_pointer_cast(device_element_variable.data()));
  T8GPU_CUDA_CHECK_LAST_ERROR();

  element_variable = device_element_variable;

  auto uniform_adaptation = [](t8_forest_t forest,
			       t8_forest_t forest_from,
			       t8_locidx_t which_tree,
			       t8_locidx_t lelement_id,
			       t8_eclass_scheme_c* ts,
			       const int is_family,
			       const int num_elements,
			       t8_element_t* elements[]) -> int { return 1; };

  t8_forest_t subgrid_forest1 {};
  t8_forest_init(&subgrid_forest1);
  t8_forest_ref(m_forest);
  t8_forest_set_adapt(subgrid_forest1, m_forest, uniform_adaptation, false);
  t8_forest_commit(subgrid_forest1);

  t8_forest_t subgrid_forest2 {};
  t8_forest_init(&subgrid_forest2);
  t8_forest_set_adapt(subgrid_forest2, subgrid_forest1, uniform_adaptation, false);
  t8_forest_commit(subgrid_forest2);


  t8_vtk_data_field_t vtk_data_field{};
  vtk_data_field.type = T8_VTK_SCALAR;
  std::strncpy(vtk_data_field.description, "density", BUFSIZ);
  if constexpr (std::is_same_v<float_type, double>) {  // no need for conversion
    vtk_data_field.data = element_variable.data();
    t8_forest_write_vtk_ext(subgrid_forest2,
                            prefix.c_str(),
                            true,  /* write_treeid */
                            true,  /* write_mpirank */
                            true,  /* write_level */
                            true,  /* write_element_id */
                            false, /* write_ghost */
                            false, /* write_curved */
                            false, /* do_not_use_API */
                            1,     /* num_data */
                            &vtk_data_field);
  } else {  // we need to convert to double precision
    thrust::host_vector<double> double_element_variable(m_num_local_elements);
    for (t8_locidx_t i = 0; i < m_num_local_elements; i++)
      double_element_variable[i] = static_cast<double>(element_variable[i]);
    vtk_data_field.data = double_element_variable.data();
    t8_forest_write_vtk_ext(subgrid_forest2,
                            prefix.c_str(),
                            true,  /* write_treeid */
                            true,  /* write_mpirank */
                            true,  /* write_level */
                            true,  /* write_element_id */
                            false, /* write_ghost */
                            false, /* write_curved */
                            false, /* do_not_use_API */
                            1,     /* num_data */
                            &vtk_data_field);
  }

  t8_forest_unref(&subgrid_forest2);
}

template<typename VariableType, typename StepType, typename SubgridType>
[[nodiscard]] t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::HostVariableInfo
t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::get_host_scalar_variable(
    t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::step_index_type     step,
    t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::variable_index_type variable,
    std::string const&                                                   name) const {
  std::unique_ptr<float_type[]> data = std::make_unique<float_type[]>(m_num_local_elements);
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(data.get(),
                                    this->get_own_variable(step, variable),
                                    sizeof(float_type) * m_num_local_elements,
                                    cudaMemcpyDeviceToHost));
  if constexpr (std::is_same_v<float_type, double>) {  // no need for type conversion.
    return {T8_VTK_SCALAR, std::move(data), name};
  } else {  // we need to cast from float to double.
    std::unique_ptr<double[]> data_double = std::make_unique<double[]>(m_num_local_elements);
    for (t8_locidx_t i = 0; i < m_num_local_elements; i++) data_double[i] = static_cast<double>(data[i]);
    return {T8_VTK_SCALAR, std::move(data_double), name};
  }
}

template<typename VariableType, typename StepType, typename SubgridType>
[[nodiscard]] t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::HostVariableInfo
t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::get_host_vector_variable(
    t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::step_index_type                    step,
    std::array<t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::variable_index_type, 3> variables,
    std::string const&                                                                  name) const {
  std::unique_ptr<float_type[]> data = std::make_unique<float_type[]>(3 * m_num_local_elements);
  for (size_t i = 0; i < 3; i++)
    T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(data.get() + i * m_num_local_elements,
                                      this->get_own_variable(step, variables[i]),
                                      sizeof(float_type) * m_num_local_elements,
                                      cudaMemcpyDeviceToHost));

  std::unique_ptr<double[]> data_interleaved = std::make_unique<double[]>(3 * m_num_local_elements);
  for (int i = 0; i < m_num_local_elements; i++) {
    for (size_t j = 0; j < 3; j++) {
      data_interleaved[3 * i + j] = static_cast<double>(data[m_num_local_elements * j + i]);
    }
  }
  return {T8_VTK_VECTOR, std::move(data_interleaved), name};
}

template<typename VariableType, typename StepType, typename SubgridType>
void t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::save_variables_to_vtk(
    std::vector<t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::HostVariableInfo> host_variables,
    std::string const&                                                             prefix) const {
  // std::vector<t8_vtk_data_field_t> vtk_data_fields(host_variables.size());
  // for (size_t i = 0; i < host_variables.size(); i++)
  //   vtk_data_fields[i] = host_variables[i].m_vtk_data_field_info_struct;

  // MPI_Barrier(m_comm);
  // t8_forest_write_vtk_ext(m_forest,
  //                         prefix.c_str(),
  //                         true,                                    /* write_treeid */
  //                         true,                                    /* write_mpirank */
  //                         true,                                    /* write_level */
  //                         true,                                    /* write_element_id */
  //                         false,                                   /* write_ghost */
  //                         false,                                   /* write_curved */
  //                         false,                                   /* do_not_use_API */
  //                         static_cast<int>(host_variables.size()), /* num_data */
  //                         vtk_data_fields.data());
}

template<typename VariableType>
__global__ void partition_data(
    int* __restrict__ ranks,
    t8_locidx_t* __restrict__ indices,
    std::array<typename t8gpu::variable_traits<VariableType>::float_type* __restrict__,
               t8gpu::variable_traits<VariableType>::nb_variables> new_variables,
    t8gpu::MemoryAccessorAll<VariableType>                         old_variables,
    typename t8gpu::variable_traits<VariableType>::float_type*     new_volume,
    typename t8gpu::variable_traits<VariableType>::float_type* const* __restrict__ old_volume,
    int num_new_elements) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_new_elements) return;

  for (size_t k = 0; k < t8gpu::variable_traits<VariableType>::nb_variables; k++) {
    new_variables[k][i] = old_variables.get(k)[ranks[i]][indices[i]];
  }

  new_volume[i] = old_volume[ranks[i]][indices[i]];
}

template<typename VariableType, typename StepType, typename SubgridType>
void t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>::partition(step_index_type step) {
  // assert(t8_forest_is_committed(m_forest));
  // t8_forest_ref(m_forest);
  // t8_forest_t partitioned_forest{};
  // t8_forest_init(&partitioned_forest);
  // t8_forest_set_partition(partitioned_forest, m_forest, true);
  // t8_forest_set_ghost(partitioned_forest, true, T8_GHOST_FACES);
  // t8_forest_commit(partitioned_forest);

  // t8_locidx_t num_old_elements{t8_forest_get_local_num_elements(m_forest)};
  // t8_locidx_t num_new_elements{t8_forest_get_local_num_elements(partitioned_forest)};

  // thrust::host_vector<t8_locidx_t> old_ranks(num_old_elements);
  // thrust::host_vector<t8_locidx_t> old_indices(num_old_elements);
  // for (t8_locidx_t i = 0; i < num_old_elements; i++) {
  //   old_ranks[i]   = m_rank;
  //   old_indices[i] = i;
  // }

  // thrust::host_vector<t8_locidx_t> new_ranks(num_new_elements);
  // thrust::host_vector<t8_locidx_t> new_indices(num_new_elements);

  // sc_array* sc_array_old_ranks_wrapper{sc_array_new_data(old_ranks.data(), sizeof(int), num_old_elements)};
  // sc_array* sc_array_old_indices_wrapper{sc_array_new_data(old_indices.data(), sizeof(t8_locidx_t), num_old_elements)};

  // sc_array* sc_array_new_ranks_wrapper{sc_array_new_data(new_ranks.data(), sizeof(int), num_new_elements)};
  // sc_array* sc_array_new_indices_wrapper{sc_array_new_data(new_indices.data(), sizeof(t8_locidx_t), num_new_elements)};

  // t8_forest_partition_data(m_forest, partitioned_forest, sc_array_old_ranks_wrapper, sc_array_new_ranks_wrapper);

  // t8_forest_partition_data(m_forest, partitioned_forest, sc_array_old_indices_wrapper, sc_array_new_indices_wrapper);

  // sc_array_destroy(sc_array_old_indices_wrapper);
  // sc_array_destroy(sc_array_new_indices_wrapper);
  // sc_array_destroy(sc_array_old_ranks_wrapper);
  // sc_array_destroy(sc_array_new_ranks_wrapper);

  // m_device_ranks   = new_ranks;
  // m_device_indices = new_indices;

  // thrust::device_vector<float_type> device_new_conserved_variables(num_new_elements *
  //                                                                  t8gpu::variable_traits<VariableType>::nb_variables);
  // thrust::device_vector<float_type> device_new_element_volume(num_new_elements);

  // std::array<float_type* __restrict__, t8gpu::variable_traits<VariableType>::nb_variables> new_variables{};
  // for (size_t k = 0; k < t8gpu::variable_traits<VariableType>::nb_variables; k++) {
  //   new_variables[k] =
  //       thrust::raw_pointer_cast(device_new_conserved_variables.data()) + sizeof(float_type) * num_new_elements;
  // }

  // constexpr int thread_block_size = 256;
  // int const     fluxes_num_blocks = (num_new_elements + thread_block_size - 1) / thread_block_size;
  // partition_data<VariableType>
  //     <<<fluxes_num_blocks, thread_block_size>>>(thrust::raw_pointer_cast(m_device_ranks.data()),
  //                                                thrust::raw_pointer_cast(m_device_indices.data()),
  //                                                new_variables,
  //                                                this->get_all_variables(step),
  //                                                thrust::raw_pointer_cast(device_new_element_volume.data()),
  //                                                this->get_all_volume(),
  //                                                num_new_elements);
  // cudaDeviceSynchronize();
  // MPI_Barrier(m_comm);

  // // resize shared and own element variables
  // this->resize(num_new_elements);

  // for (int k = 0; k < t8gpu::variable_traits<VariableType>::nb_variables; k++) {
  //   this->set_variable(step, static_cast<VariableType>(k), new_variables[k]);
  // }
  // this->set_volume(std::move(device_new_element_volume));

  // UserData* forest_user_data = static_cast<UserData*>(t8_forest_get_user_data(m_forest));
  // t8_forest_set_user_data(partitioned_forest, forest_user_data);
  // t8_forest_unref(&m_forest);
  // m_forest = partitioned_forest;

  // m_num_ghost_elements = t8_forest_get_num_ghosts(m_forest);
  // m_num_local_elements = t8_forest_get_local_num_elements(m_forest);
}
