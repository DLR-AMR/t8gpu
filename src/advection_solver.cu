#include <advection_solver.h>
#include <t8.h>
#include <t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_examples.h>
#include <t8_forest/t8_forest.h>
#include <t8_forest/t8_forest_iterate.h>
#include <t8_forest/t8_forest_partition.h>
#include <utils/cuda.h>
#include <utils/profiling.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <t8_schemes/t8_default/t8_default_cxx.hxx>

struct forest_user_data_t {
  thrust::host_vector<double>* element_refinement_criteria;
};

static int adapt_callback_initialization(t8_forest_t forest, t8_forest_t forest_from, t8_locidx_t which_tree, t8_locidx_t lelement_id,
					 t8_eclass_scheme_c* ts, const int is_family, const int num_elements, t8_element_t* elements[]);

static int adapt_callback_iteration(t8_forest_t forest, t8_forest_t forest_from, t8_locidx_t which_tree, t8_locidx_t lelement_id, t8_eclass_scheme_c* ts,
				    const int is_family, const int num_elements, t8_element_t* elements[]);

__global__ static void compute_refinement_criteria(double const* __restrict__ variable, double* __restrict__ criteria, int nb_elements);

__global__ static void adapt_variable_and_volume(double const* __restrict__ variable_old, double const* __restrict__ volume_old,
                                                 double* __restrict__ variable_new, double* __restrict__ volume_new, t8_locidx_t* adapt_data,
                                                 int nb_new_elements);

__global__ void partition_data(int* __restrict__ ranks, t8_locidx_t* __restrict__ indices,
			       double* __restrict__ new_variable, double* __restrict__ new_volume,
			       double const*const* __restrict__ old_variable, double const*const* __restrict__ old_volume,
			       int num_new_elements);

__global__ static void compute_fluxes(double** __restrict__ variables, double** __restrict__ fluxes, double const* __restrict__ normal,
                                      double const* __restrict__ area, int const* e_idx, int* rank, t8_locidx_t* indices, int nb_edges);

__global__ static void explicit_euler_time_step(double const* __restrict__ variable_prev, double* __restrict__ variable_next,
                                                double const* __restrict__ volume, double* __restrict__ fluxes, double delta_t, int nb_elements);

t8gpu::AdvectionSolver::AdvectionSolver(sc_MPI_Comm comm)
    : comm(comm),
      cmesh(t8_cmesh_new_periodic_hybrid(comm)),
      scheme(t8_scheme_new_default_cxx()),
      forest(t8_forest_new_uniform(cmesh, scheme, 6, true, comm)),
      delta_t(0.5 * std::pow(0.5, max_level) / sqrt(2.0)) {
  t8_forest_t new_forest {};
  t8_forest_init(&new_forest);
  t8_forest_set_adapt(new_forest, forest, adapt_callback_initialization, true);
  t8_forest_set_ghost(new_forest, true, T8_GHOST_FACES);
  t8_forest_set_balance(new_forest, forest, false);
  t8_forest_set_partition(new_forest, forest, true);
  t8_forest_commit(new_forest);
  forest = new_forest;

  MPI_Comm_size(comm, &nb_ranks);
  MPI_Comm_rank(comm, &rank);

  num_ghost_elements = t8_forest_get_num_ghosts(forest);
  num_local_elements = t8_forest_get_local_num_elements(forest);

  ranks.resize(num_local_elements + num_ghost_elements);
  indices.resize(num_local_elements + num_ghost_elements);
  for (t8_locidx_t i=0; i<num_local_elements; i++) {
    ranks[i] = rank;
    indices[i] = i;
  }
  sc_array* sc_array_ranks_wrapper {sc_array_new_data(ranks.data(), sizeof(int), num_local_elements + num_ghost_elements)};
  t8_forest_ghost_exchange_data(forest, sc_array_ranks_wrapper);
  sc_array_destroy(sc_array_ranks_wrapper);

  sc_array* sc_array_indices_wrapper {sc_array_new_data(indices.data(), sizeof(t8_locidx_t), num_local_elements + num_ghost_elements)};
  t8_forest_ghost_exchange_data(forest, sc_array_indices_wrapper);
  sc_array_destroy(sc_array_indices_wrapper);

  device_ranks = ranks;
  device_indices = indices;

  thrust::host_vector<double> element_variable(num_local_elements);
  thrust::host_vector<double> element_volume(num_local_elements);
  element_refinement_criteria.resize(num_local_elements);

  t8_locidx_t num_local_trees = t8_forest_get_num_local_trees(forest);
  t8_locidx_t element_idx = 0;
  for (t8_locidx_t tree_idx = 0; tree_idx < num_local_trees; tree_idx++) {
    t8_eclass_t tree_class {t8_forest_get_tree_class(forest, tree_idx)};
    t8_eclass_scheme_c* eclass_scheme {t8_forest_get_eclass_scheme(forest, tree_class)};

    t8_locidx_t num_elements_in_tree {t8_forest_get_tree_num_elements(forest, tree_idx)};
    for (t8_locidx_t tree_element_idx = 0; tree_element_idx < num_elements_in_tree; tree_element_idx++) {
      const t8_element_t* element {t8_forest_get_element_in_tree(forest, tree_idx, tree_element_idx)};

      double center[3];
      t8_forest_element_centroid(forest, tree_idx, element, center);

      element_variable[element_idx] = sqrt((0.5 - center[0]) * (0.5 - center[0]) + (0.5 - center[1]) * (0.5 - center[1])) - 0.25;
      element_volume[element_idx] = t8_forest_element_volume(forest, tree_idx, element);

      element_idx++;
    }
  }

  device_element_variable_prev.resize(num_local_elements);
  device_element_fluxes.resize(num_local_elements);
  device_element_refinement_criteria.resize(num_local_elements);

  device_element_variable_next = element_variable;
  device_element_volume = element_volume;
  double* device_element_fluxes_ptr {device_element_fluxes.get_own()};
  T8GPU_CUDA_CHECK_ERROR(cudaMemset(device_element_fluxes_ptr, 0, sizeof(double)*num_local_elements));

  compute_edge_connectivity();
  device_face_neighbors = face_neighbors;
  device_face_normals = face_normals;
  device_face_area = face_area;

  // TODO: remove allocation out of RAII paradigm
  forest_user_data_t* forest_user_data = static_cast<forest_user_data_t*>(malloc(sizeof(forest_user_data_t)));
  assert(forest_user_data != nullptr);

  forest_user_data->element_refinement_criteria = &element_refinement_criteria;
  t8_forest_set_user_data(forest, forest_user_data);
}

t8gpu::AdvectionSolver::~AdvectionSolver() {
  forest_user_data_t* forest_user_data {static_cast<forest_user_data_t*>(t8_forest_get_user_data(forest))};
  free(forest_user_data);

  t8_forest_unref(&forest);
  t8_cmesh_destroy(&cmesh);
}

void t8gpu::AdvectionSolver::iterate() {
  std::swap(device_element_variable_next, device_element_variable_prev);

  double* device_element_fluxes_ptr {device_element_fluxes.get_own()};
  T8GPU_CUDA_CHECK_ERROR(cudaMemset(device_element_fluxes_ptr, 0, sizeof(double)*device_element_fluxes.size()));

  cudaDeviceSynchronize();
  MPI_Barrier(comm);
  constexpr int thread_block_size = 256;
  const int fluxes_num_blocks = (num_local_faces + thread_block_size - 1) / thread_block_size;
  compute_fluxes<<<fluxes_num_blocks, thread_block_size>>>(
							   device_element_variable_prev.get_all(),
							   device_element_fluxes.get_all(),
							   thrust::raw_pointer_cast(device_face_normals.data()),
							   thrust::raw_pointer_cast(device_face_area.data()),
							   thrust::raw_pointer_cast(device_face_neighbors.data()),
							   thrust::raw_pointer_cast(device_ranks.data()),
							   thrust::raw_pointer_cast(device_indices.data()),
							   num_local_faces);
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(comm);

  const int euler_num_blocks = (t8_forest_get_local_num_elements(forest) + thread_block_size - 1) / thread_block_size;
  explicit_euler_time_step<<<euler_num_blocks, thread_block_size>>>(
      device_element_variable_prev.get_own(), device_element_variable_next.get_own(),
      device_element_volume.get_own(), device_element_fluxes.get_own(), delta_t, num_local_elements);
  T8GPU_CUDA_CHECK_LAST_ERROR();
}

void t8gpu::AdvectionSolver::adapt() {
  constexpr int thread_block_size = 256;
  const int fluxes_num_blocks = (num_local_elements + thread_block_size - 1) / thread_block_size;
  compute_refinement_criteria<<<fluxes_num_blocks, thread_block_size>>>(device_element_variable_next.get_own(),
                                                                        thrust::raw_pointer_cast(device_element_refinement_criteria.data()),
                                                                        num_local_elements);
  T8GPU_CUDA_CHECK_LAST_ERROR();

  element_refinement_criteria = device_element_refinement_criteria;

  t8_forest_ref(forest);
  assert(t8_forest_is_committed(forest));

  t8_forest_t adapted_forest {};
  t8_forest_init(&adapted_forest);
  t8_forest_set_adapt(adapted_forest, forest, adapt_callback_iteration, false);
  t8_forest_set_ghost(adapted_forest, true, T8_GHOST_FACES);
  t8_forest_set_balance(adapted_forest, forest, true);
  t8_forest_commit(adapted_forest);

  t8_locidx_t old_idx = 0;
  t8_locidx_t new_idx = 0;

  t8_locidx_t num_new_elements {t8_forest_get_local_num_elements(adapted_forest)};
  t8_locidx_t num_old_elements {t8_forest_get_local_num_elements(forest)};

  thrust::host_vector<double> adapted_element_variable(num_new_elements);
  thrust::host_vector<double> adapted_element_volume(num_new_elements);
  thrust::host_vector<t8_locidx_t> element_adapt_data(num_new_elements + 1);

  thrust::host_vector<t8_locidx_t> old_levels(num_old_elements);
  thrust::host_vector<t8_locidx_t> new_levels(num_new_elements);

  t8_locidx_t num_old_local_trees = {t8_forest_get_num_local_trees(forest)};
  t8_locidx_t num_new_local_trees = {t8_forest_get_num_local_trees(forest)};

  t8_locidx_t current_idx = 0;
  for (t8_locidx_t tree_idx = 0; tree_idx < num_old_local_trees; tree_idx++) {
    t8_eclass_t old_tree_class {t8_forest_get_tree_class(forest, tree_idx)};
    t8_eclass_scheme_c* old_scheme = {t8_forest_get_eclass_scheme(forest, old_tree_class)};

    t8_locidx_t num_elements_in_tree {t8_forest_get_tree_num_elements(forest, tree_idx)};

    for (t8_locidx_t elem_idx = 0; elem_idx < num_elements_in_tree; elem_idx++) {
      t8_element_t const* element {t8_forest_get_element_in_tree(forest, tree_idx, elem_idx)};
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

  element_refinement_criteria.resize(num_new_elements);

  forest_user_data_t* forest_user_data {static_cast<forest_user_data_t*>(t8_forest_get_user_data(forest))};
  assert(forest_user_data != nullptr);

  t8_forest_set_user_data(adapted_forest, forest_user_data);
  t8_forest_unref(&forest);

  thrust::device_vector<double> device_element_variable_next_adapted(num_new_elements);
  thrust::device_vector<double> device_element_volume_adapted(num_new_elements);
  t8_locidx_t* device_element_adapt_data {};
  T8GPU_CUDA_CHECK_ERROR(cudaMalloc(&device_element_adapt_data, (num_new_elements + 1) * sizeof(t8_locidx_t)));
  T8GPU_CUDA_CHECK_ERROR(
      cudaMemcpy(device_element_adapt_data, element_adapt_data.data(), element_adapt_data.size() * sizeof(t8_locidx_t), cudaMemcpyHostToDevice));
  const int adapt_num_blocks = (num_new_elements + thread_block_size - 1) / thread_block_size;
  adapt_variable_and_volume<<<adapt_num_blocks, thread_block_size>>>(
      device_element_variable_next.get_own(), device_element_volume.get_own(),
      thrust::raw_pointer_cast(device_element_variable_next_adapted.data()), thrust::raw_pointer_cast(device_element_volume_adapted.data()),
      device_element_adapt_data, num_new_elements);
  T8GPU_CUDA_CHECK_LAST_ERROR();
  T8GPU_CUDA_CHECK_ERROR(cudaFree(device_element_adapt_data));
  device_element_variable_next = std::move(device_element_variable_next_adapted);
  device_element_volume = std::move(device_element_volume_adapted);

  forest = adapted_forest;

  device_element_variable_prev.resize(num_new_elements);

  device_element_fluxes.resize(num_new_elements);
  double* device_element_fluxes_ptr {device_element_fluxes.get_own()};
  T8GPU_CUDA_CHECK_ERROR(cudaMemset(device_element_fluxes_ptr, 0, sizeof(double)*num_new_elements));

  device_element_refinement_criteria.resize(num_new_elements);

  num_ghost_elements = t8_forest_get_num_ghosts(forest);
  num_local_elements = t8_forest_get_local_num_elements(forest);
}

void t8gpu::AdvectionSolver::partition() {
  assert(t8_forest_is_committed(forest));
  t8_forest_ref(forest);
  t8_forest_t partitioned_forest {};
  t8_forest_init(&partitioned_forest);
  t8_forest_set_partition(partitioned_forest, forest, true);
  t8_forest_set_ghost(partitioned_forest, true, T8_GHOST_FACES);
  t8_forest_commit(partitioned_forest);

  t8_locidx_t num_old_elements {t8_forest_get_local_num_elements(forest)};
  t8_locidx_t num_new_elements {t8_forest_get_local_num_elements(partitioned_forest)};

  thrust::host_vector<t8_locidx_t> old_ranks(num_old_elements);
  thrust::host_vector<t8_locidx_t> old_indices(num_old_elements);
  for (t8_locidx_t i=0; i<num_old_elements; i++) {
    old_ranks[i] = rank;
    old_indices[i] = i;
  }

  thrust::host_vector<t8_locidx_t> new_ranks(num_new_elements);
  thrust::host_vector<t8_locidx_t> new_indices(num_new_elements);

  sc_array* sc_array_old_ranks_wrapper {sc_array_new_data(old_ranks.data(), sizeof(int), num_old_elements)};
  sc_array* sc_array_old_indices_wrapper {sc_array_new_data(old_indices.data(), sizeof(t8_locidx_t), num_old_elements)};

  sc_array* sc_array_new_ranks_wrapper {sc_array_new_data(new_ranks.data(), sizeof(int), num_new_elements)};
  sc_array* sc_array_new_indices_wrapper {sc_array_new_data(new_indices.data(), sizeof(t8_locidx_t), num_new_elements)};

  t8_forest_partition_data(forest, partitioned_forest,
			   sc_array_old_ranks_wrapper,
			   sc_array_new_ranks_wrapper);

  t8_forest_partition_data(forest, partitioned_forest,
			   sc_array_old_indices_wrapper,
			   sc_array_new_indices_wrapper);

  sc_array_destroy(sc_array_old_indices_wrapper);
  sc_array_destroy(sc_array_new_indices_wrapper);
  sc_array_destroy(sc_array_old_ranks_wrapper);
  sc_array_destroy(sc_array_new_ranks_wrapper);

  thrust::device_vector<int> device_new_ranks = new_ranks;
  thrust::device_vector<t8_locidx_t> device_new_indices = new_indices;

  t8gpu::SharedDeviceVector<double> device_new_element_variable(num_new_elements);
  t8gpu::SharedDeviceVector<double> device_new_element_volume(num_new_elements);

  constexpr int thread_block_size = 256;
  const int fluxes_num_blocks = (num_new_elements + thread_block_size - 1) / thread_block_size;
  partition_data<<<fluxes_num_blocks, thread_block_size>>>(thrust::raw_pointer_cast(device_new_ranks.data()),
							   thrust::raw_pointer_cast(device_new_indices.data()),
							   device_new_element_variable.get_own(),
							   device_new_element_volume.get_own(),
							   device_element_variable_next.get_all(),
							   device_element_volume.get_all(),
							   num_new_elements);
  cudaDeviceSynchronize();
  MPI_Barrier(comm);

  device_element_variable_next = std::move(device_new_element_variable);
  device_element_variable_prev.resize(num_new_elements);
  device_element_volume = std::move(device_new_element_volume);
  device_element_refinement_criteria.resize(num_new_elements);
  device_element_fluxes.resize(num_new_elements);

  forest_user_data_t* forest_user_data = static_cast<forest_user_data_t*>(t8_forest_get_user_data(forest));
  t8_forest_set_user_data(partitioned_forest, forest_user_data);
  t8_forest_unref(&forest);
  forest = partitioned_forest;

  num_ghost_elements = t8_forest_get_num_ghosts(forest);
  num_local_elements = t8_forest_get_local_num_elements(forest);
}

void t8gpu::AdvectionSolver::compute_connectivity_information() {
  t8_locidx_t num_ghost_elements {t8_forest_get_num_ghosts(forest)};
  t8_locidx_t num_local_elements {t8_forest_get_local_num_elements(forest)};

  ranks.resize(num_local_elements + num_ghost_elements);
  indices.resize(num_local_elements + num_ghost_elements);
  for (t8_locidx_t i=0; i<num_local_elements; i++) {
    ranks[i] = rank;
    indices[i] = i;
  }
  sc_array* sc_array_ranks_wrapper {sc_array_new_data(ranks.data(), sizeof(int), num_local_elements + num_ghost_elements)};
  t8_forest_ghost_exchange_data(forest, sc_array_ranks_wrapper);
  sc_array_destroy(sc_array_ranks_wrapper);

  sc_array* sc_array_indices_wrapper {sc_array_new_data(indices.data(), sizeof(t8_locidx_t), num_local_elements + num_ghost_elements)};
  t8_forest_ghost_exchange_data(forest, sc_array_indices_wrapper);
  sc_array_destroy(sc_array_indices_wrapper);

  device_ranks = ranks;
  device_indices = indices;

  compute_edge_connectivity();
  device_face_neighbors = face_neighbors;
  device_face_normals = face_normals;
  device_face_area = face_area;
}

void t8gpu::AdvectionSolver::save_vtk(const std::string& prefix) const {
  thrust::host_vector<double> element_variable(num_local_elements);
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(element_variable.data(), device_element_variable_next.get_own(), sizeof(double)*element_variable.size(), cudaMemcpyDeviceToHost));

  t8_vtk_data_field_t vtk_data_field {};
  vtk_data_field.type = T8_VTK_SCALAR;
  strcpy(vtk_data_field.description, "advection variable");
  vtk_data_field.data = element_variable.data();
  t8_forest_write_vtk_ext(forest, prefix.c_str(), 1, 1, 1, 1, 0, 0, 0, 1, &vtk_data_field);
}

double t8gpu::AdvectionSolver::compute_integral() const {
  double local_integral = 0.0;
  double const* mem {device_element_variable_next.get_own()};
  thrust::host_vector<double> variable(num_local_elements);
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(variable.data(), mem, sizeof(double)*device_element_variable_next.size(), cudaMemcpyDeviceToHost));
  thrust::host_vector<double> volume(num_local_elements);
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(volume.data(), device_element_volume.get_own(), sizeof(double)*device_element_volume.size(), cudaMemcpyDeviceToHost));

  for (t8_locidx_t i=0; i<num_local_elements; i++) {
    local_integral += volume[i] * variable[i];
  }
  double global_integral {};
  MPI_Allreduce(&local_integral, &global_integral, 1, MPI_DOUBLE, MPI_SUM, comm);
  return global_integral;
}

void t8gpu::AdvectionSolver::compute_edge_connectivity() {
  face_neighbors.clear();
  face_normals.clear();
  face_area.clear();

  assert(t8_forest_is_committed(forest));
  t8_locidx_t num_local_elements {t8_forest_get_local_num_elements(forest)};

  t8_locidx_t num_local_trees {t8_forest_get_num_local_trees(forest)};
  t8_locidx_t element_idx = 0;
  for (t8_locidx_t tree_idx = 0; tree_idx < num_local_trees; tree_idx++) {
    t8_eclass_t tree_class = t8_forest_get_tree_class(forest, tree_idx);
    t8_eclass_scheme_c* eclass_scheme {t8_forest_get_eclass_scheme(forest, tree_class)};

    t8_locidx_t num_elements_in_tree {t8_forest_get_tree_num_elements(forest, tree_idx)};
    for (t8_locidx_t tree_element_idx = 0; tree_element_idx < num_elements_in_tree; tree_element_idx++) {
      const t8_element_t* element {t8_forest_get_element_in_tree(forest, tree_idx, tree_element_idx)};

      t8_locidx_t num_faces {eclass_scheme->t8_element_num_faces(element)};
      for (t8_locidx_t face_idx = 0; face_idx < num_faces; face_idx++) {
        int num_neighbors {};
        int* dual_faces {};
        t8_locidx_t* neighbor_ids {};
        t8_element_t** neighbors {};
        t8_eclass_scheme_c* neigh_scheme {};

        t8_forest_leaf_face_neighbors(forest, tree_idx, element, &neighbors, face_idx, &dual_faces, &num_neighbors, &neighbor_ids, &neigh_scheme,
                                      true);

	for (int i=0; i<num_neighbors; i++) {
	  if (neighbor_ids[i] >= num_local_elements && rank < ranks[neighbor_ids[i]]) {
	    face_neighbors.push_back(element_idx);
	    face_neighbors.push_back(neighbor_ids[i]);
	    double face_normal[3];
	    t8_forest_element_face_normal(forest, tree_idx, element, face_idx, face_normal);
	    face_normals.push_back(face_normal[0]);
	    face_normals.push_back(face_normal[1]);
	    face_area.push_back(t8_forest_element_face_area(forest, tree_idx, element, face_idx) / static_cast<double>(num_neighbors));
	  }
	}

        if ((num_neighbors == 1) && (neighbor_ids[0] < num_local_elements) &&
            ((neighbor_ids[0] > element_idx) ||
             (neighbor_ids[0] < element_idx && neigh_scheme[0].t8_element_level(neighbors[0]) < eclass_scheme->t8_element_level(element)))) {
	  face_neighbors.push_back(element_idx);
	  face_neighbors.push_back(neighbor_ids[0]);
	  double face_normal[3];
	  t8_forest_element_face_normal(forest, tree_idx, element, face_idx, face_normal);
	  face_normals.push_back(face_normal[0]);
	  face_normals.push_back(face_normal[1]);
	  face_area.push_back(t8_forest_element_face_area(forest, tree_idx, element, face_idx));
        }
        T8_FREE(neighbors);
        T8_FREE(dual_faces);
        T8_FREE(neighbor_ids);
      }

      element_idx++;
    }
  }

  num_local_faces = face_area.size();
}

static int adapt_callback_initialization(t8_forest_t forest, t8_forest_t forest_from, t8_locidx_t which_tree, t8_locidx_t lelement_id,
					 t8_eclass_scheme_c* ts, const int is_family, const int num_elements, t8_element_t* elements[]) {
  t8_locidx_t element_level {ts->t8_element_level(elements[0])};

  double b = 0.02;

  if (element_level < t8gpu::AdvectionSolver::max_level) {
    double center[3];
    t8_forest_element_centroid(forest_from, which_tree, elements[0], center);

    double variable = sqrt((0.5 - center[0]) * (0.5 - center[0]) + (0.5 - center[1]) * (0.5 - center[1])) - 0.25;

    if (std::abs(variable) < b) return 1;
  }
  if (element_level > t8gpu::AdvectionSolver::min_level && is_family) {
    double center[] = {0.0, 0.0, 0.0};
    double current_element_center[] = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < 4; i++) {
      t8_forest_element_centroid(forest_from, which_tree, elements[i], current_element_center);
      for (size_t j = 0; j < 3; j++) {
        center[j] += current_element_center[j] / 4.0;
      }
    }

    double variable = sqrt((0.5 - center[0]) * (0.5 - center[0]) + (0.5 - center[1]) * (0.5 - center[1])) - 0.25;

    if (std::abs(variable) > b) return -1;
  }

  return 0;
}

static int adapt_callback_iteration(t8_forest_t forest, t8_forest_t forest_from, t8_locidx_t which_tree, t8_locidx_t lelement_id, t8_eclass_scheme_c* ts,
				    const int is_family, const int num_elements, t8_element_t* elements[]) {
  forest_user_data_t* forest_user_data = static_cast<forest_user_data_t*>(t8_forest_get_user_data(forest_from));
  assert(forest_user_data != nullptr);

  t8_locidx_t element_level {ts->t8_element_level(elements[0])};

  t8_locidx_t tree_offset = t8_forest_get_tree_element_offset(forest_from, which_tree);

  double b = 1.0;
  double h = std::pow(0.5, element_level);

  if (element_level < t8gpu::AdvectionSolver::max_level) {
    double variable = (*forest_user_data->element_refinement_criteria)[tree_offset + lelement_id];

    if (std::abs(variable) < b * h) {
      return 1;
    }
  }
  if (element_level > t8gpu::AdvectionSolver::min_level && is_family) {
    double variable = 0.0;
    for (size_t i = 0; i < 4; i++) {
      variable += (*forest_user_data->element_refinement_criteria)[tree_offset + lelement_id + i] / 4.0;
    }

    if (std::abs(variable) > (2 * h) * b) {
      return -1;
    }
  }

  return 0;
}

__global__ static void compute_refinement_criteria(double const* __restrict__ variable, double* __restrict__ criteria, int nb_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_elements) return;

  criteria[i] = variable[i];
}

__global__ static void adapt_variable_and_volume(double const* __restrict__ variable_old, double const* __restrict__ volume_old,
                                                 double* __restrict__ variable_new, double* __restrict__ volume_new, t8_locidx_t* adapt_data,
                                                 int nb_new_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_new_elements) return;

  int diff = adapt_data[i + 1] - adapt_data[i];
  int nb_elements_sum = max(1, diff);

  volume_new[i] = volume_old[adapt_data[i]] * ((diff == 0 ? 0.25 : (diff == 1 ? 1.0 : 4.0)));
  if (i > 0 && adapt_data[i - 1] == adapt_data[i]) {
    volume_new[i] = volume_old[adapt_data[i]] * 0.25;
  }

  variable_new[i] = 0.0;
  for (int j = 0; j < nb_elements_sum; j++) {
    variable_new[i] += variable_old[adapt_data[i] + j] / static_cast<double>(nb_elements_sum);
  }
}

__global__ void partition_data(int* __restrict__ ranks, t8_locidx_t* __restrict__ indices,
			       double* __restrict__ new_variable, double* __restrict__ new_volume,
			       double const*const* __restrict__ old_variable, double const*const* __restrict__ old_volume,
			       int num_new_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_new_elements) return;

  new_variable[i] = old_variable[ranks[i]][indices[i]];
  new_volume[i] = old_volume[ranks[i]][indices[i]];
}

__global__ static void compute_fluxes(double** __restrict__ variables, double** __restrict__ fluxes, double const* __restrict__ normal,
                                      double const* __restrict__ area, int const* e_idx, int* rank, t8_locidx_t* indices, int nb_edges) {
  double a[2] = {0.5 * sqrt(2.0), 0.5 * sqrt(2.0)};

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_edges) return;

  double flux = area[i] * (a[0] * normal[2 * i] + a[1] * normal[2 * i + 1]);

  if (flux > 0.0) {
    flux *= variables[rank[e_idx[2 * i]]][indices[e_idx[2 * i]]];
  } else {
    flux *= variables[rank[e_idx[2 * i + 1]]][indices[e_idx[2 * i + 1]]];
  }

  atomicAdd(&fluxes[rank[e_idx[2 * i]]][indices[e_idx[2 * i]]], -flux);
  atomicAdd(&fluxes[rank[e_idx[2 * i + 1]]][indices[e_idx[2 * i + 1]]], flux);
}

__global__ static void explicit_euler_time_step(double const* __restrict__ variable_prev, double* __restrict__ variable_next,
                                                double const* __restrict__ volume, double* __restrict__ fluxes, double delta_t, int nb_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_elements) return;

  variable_next[i] = variable_prev[i] + delta_t / volume[i] * fluxes[i];
}
