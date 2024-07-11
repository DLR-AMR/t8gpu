#include <advection_solver.h>
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
#include <t8_schemes/t8_default/t8_default_cxx.hxx>

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

using float_type = t8gpu::AdvectionSolver::float_type;

struct forest_user_data_t {
  thrust::host_vector<float_type>* element_refinement_criteria;
};

static int adapt_callback_initialization(t8_forest_t forest, t8_forest_t forest_from, t8_locidx_t which_tree, t8_locidx_t lelement_id,
					 t8_eclass_scheme_c* ts, const int is_family, const int num_elements, t8_element_t* elements[]);

static int adapt_callback_iteration(t8_forest_t forest, t8_forest_t forest_from, t8_locidx_t which_tree, t8_locidx_t lelement_id, t8_eclass_scheme_c* ts,
				    const int is_family, const int num_elements, t8_element_t* elements[]);

__global__ static void compute_refinement_criteria(float_type const* __restrict__ rho_v1,
						   float_type const* __restrict__ rho_v2,
						   float_type* __restrict__ criteria, int nb_elements);

__global__ static void adapt_variables_and_volume(float_type const* __restrict__ rho_old,
						  float_type const* __restrict__ rho_v1_old,
						  float_type const* __restrict__ rho_v2_old,
						  float_type const* __restrict__ rho_e_old,
						  float_type const* __restrict__ volume_old,
						  float_type* __restrict__ rho_new,
						  float_type* __restrict__ rho_v1_new,
						  float_type* __restrict__ rho_v2_new,
						  float_type* __restrict__ rho_e_new,
						  float_type* __restrict__ volume_new,
						  t8_locidx_t* adapt_data,
						  int nb_new_elements);

__global__ void partition_data(int* __restrict__ ranks, t8_locidx_t* __restrict__ indices,
			       float_type* __restrict__ new_rho,
			       float_type* __restrict__ new_rho_v1,
			       float_type* __restrict__ new_rho_v2,
			       float_type* __restrict__ new_rho_e,
			       float_type* __restrict__ new_volume,
			       float_type const*const* __restrict__ old_rho,
			       float_type const*const* __restrict__ old_rho_v1,
			       float_type const*const* __restrict__ old_rho_v2,
			       float_type const*const* __restrict__ old_rho_e,
			       float_type const*const* __restrict__ old_volume,
			       int num_new_elements);

__global__ static void hll_compute_fluxes(float_type** __restrict__ rho,
					  float_type** __restrict__ rho_v1,
					  float_type** __restrict__ rho_v2,
					  float_type** __restrict__ rho_e,
					  float_type** __restrict__ rho_fluxes,
					  float_type** __restrict__ rho_v1_fluxes,
					  float_type** __restrict__ rho_v2_fluxes,
					  float_type** __restrict__ rho_e_fluxes,
					  float_type* __restrict__ speed_estimate,
					  float_type const* __restrict__ normal,
					  float_type const* __restrict__ area,
					  int const* e_idx, int* rank,
					  t8_locidx_t* indices, int nb_edges);

__global__ static void kepes_compute_fluxes(float_type** __restrict__ rho,
					    float_type** __restrict__ rho_v1,
					    float_type** __restrict__ rho_v2,
					    float_type** __restrict__ rho_e,
					    float_type** __restrict__ rho_fluxes,
					    float_type** __restrict__ rho_v1_fluxes,
					    float_type** __restrict__ rho_v2_fluxes,
					    float_type** __restrict__ rho_e_fluxes,
					    float_type* __restrict__ speed_estimates,
					    float_type const* __restrict__ normal,
					    float_type const* __restrict__ area,
					    int const* e_idx, int* rank,
					    t8_locidx_t* indices, int nb_edges);

t8gpu::AdvectionSolver::AdvectionSolver(sc_MPI_Comm comm)
    : comm(comm),
      cmesh(t8_cmesh_new_periodic(comm, 2)),
      scheme(t8_scheme_new_default_cxx()),
      forest(t8_forest_new_uniform(cmesh, scheme, 7, true, comm)) {
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

  rho_prev    = rho_0;
  rho_v1_prev = rho_v1_0;
  rho_v2_prev = rho_v2_0;
  rho_e_prev  = rho_e_0;

  rho_next    = rho_3;
  rho_v1_next = rho_v1_3;
  rho_v2_next = rho_v2_3;
  rho_e_next  = rho_e_3;

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

  thrust::host_vector<float_type> element_rho(num_local_elements);
  thrust::host_vector<float_type> element_rho_v1(num_local_elements);
  thrust::host_vector<float_type> element_rho_v2(num_local_elements);
  thrust::host_vector<float_type> element_rho_e(num_local_elements);

  thrust::host_vector<float_type> element_volume(num_local_elements);

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

      float_type sigma = float_type{0.05}/sqrt(float_type{2.0});

      double x = center[0]-0.5;
      double y = center[1]-0.5;

      float_type rho = static_cast<float_type>(std::abs(y) < 0.25 ? 2.0 : 1.0);

      float_type v1 = static_cast<float_type>(std::abs(y) < 0.25 ? -0.5 : 0.5);
      float_type v2 = static_cast<float_type>(0.1*sin(4.0*M_PI*x)*(exp(-((y-0.25)/(2*sigma))*((y-0.25)/(2*sigma)))+exp(-((y+0.25)/(2*sigma))*((y+0.25)/(2*sigma)))));

      float_type rho_v1 = rho*v1;
      float_type rho_v2 = rho*v2;

      element_rho[element_idx]    = rho;
      element_rho_v1[element_idx] = rho_v1;
      element_rho_v2[element_idx] = rho_v2;
      element_rho_e[element_idx]  = float_type{2.0}/(gamma-float_type{1.0}) + float_type{0.5}*(rho_v1 * rho_v1 + rho_v2 * rho_v2) / rho;


      element_volume[element_idx] = static_cast<float_type>(t8_forest_element_volume(forest, tree_idx, element));

      element_idx++;
    }
  }

  // resize shared and owned element variables
  device_element.resize(num_local_elements);

  element_refinement_criteria.resize(num_local_elements);
  device_element_refinement_criteria.resize(num_local_elements);

  // copy new shared element variables
  device_element.copy(rho_next, element_rho);
  device_element.copy(rho_v1_next, element_rho_v1);
  device_element.copy(rho_v2_next, element_rho_v2);
  device_element.copy(rho_e_next, element_rho_e);

  // device_element[volume] = element_volume;
  device_element.copy(volume, element_volume);

  // fill fluxes device element variable
  float_type* device_element_fluxes_ptr {device_element.get_own(rho_fluxes)};
  T8GPU_CUDA_CHECK_ERROR(cudaMemset(device_element_fluxes_ptr, 0, 4*sizeof(float_type)*num_local_elements));

  compute_edge_connectivity();
  device_face_neighbors = face_neighbors;
  device_face_normals = face_normals;
  device_face_area = face_area;
  device_face_speed_estimate.resize(device_face_area.size());
  thrust::fill(device_face_speed_estimate.begin(),
	       device_face_speed_estimate.end(),
	       std::numeric_limits<float_type>::infinity());

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

void t8gpu::AdvectionSolver::iterate(float_type delta_t) {
  std::swap(rho_prev, rho_next);
  std::swap(rho_v1_prev, rho_v1_next);
  std::swap(rho_v2_prev, rho_v2_next);
  std::swap(rho_e_prev, rho_e_next);

  compute_fluxes(rho_prev,
		 rho_v1_prev,
		 rho_v2_prev,
		 rho_e_prev);

  constexpr int thread_block_size = 256;
  const int SSP_num_blocks = (num_local_elements + thread_block_size - 1) / thread_block_size;
  t8gpu::timestepping::SSP_3RK_step1<<<SSP_num_blocks, thread_block_size>>>(
      device_element.get_own(rho_prev),
      device_element.get_own(rho_v1_prev),
      device_element.get_own(rho_v2_prev),
      device_element.get_own(rho_e_prev),
      device_element.get_own(rho_1),
      device_element.get_own(rho_v1_1),
      device_element.get_own(rho_v2_1),
      device_element.get_own(rho_e_1),
      device_element.get_own(volume),
      device_element.get_own(rho_fluxes),
      device_element.get_own(rho_v1_fluxes),
      device_element.get_own(rho_v2_fluxes),
      device_element.get_own(rho_e_fluxes),
      delta_t, num_local_elements);
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(comm);

  compute_fluxes(rho_1,
		 rho_v1_1,
		 rho_v2_1,
		 rho_e_1);

  t8gpu::timestepping::SSP_3RK_step2<<<SSP_num_blocks, thread_block_size>>>(
      device_element.get_own(rho_prev),
      device_element.get_own(rho_v1_prev),
      device_element.get_own(rho_v2_prev),
      device_element.get_own(rho_e_prev),
      device_element.get_own(rho_1),
      device_element.get_own(rho_v1_1),
      device_element.get_own(rho_v2_1),
      device_element.get_own(rho_e_1),
      device_element.get_own(rho_2),
      device_element.get_own(rho_v1_2),
      device_element.get_own(rho_v2_2),
      device_element.get_own(rho_e_2),
      device_element.get_own(volume),
      device_element.get_own(rho_fluxes),
      device_element.get_own(rho_v1_fluxes),
      device_element.get_own(rho_v2_fluxes),
      device_element.get_own(rho_e_fluxes),
      delta_t, num_local_elements);
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(comm);

  compute_fluxes(rho_2,
		 rho_v1_2,
		 rho_v2_2,
		 rho_e_2);

  t8gpu::timestepping::SSP_3RK_step3<<<SSP_num_blocks, thread_block_size>>>(
      device_element.get_own(rho_prev),
      device_element.get_own(rho_v1_prev),
      device_element.get_own(rho_v2_prev),
      device_element.get_own(rho_e_prev),
      device_element.get_own(rho_2),
      device_element.get_own(rho_v1_2),
      device_element.get_own(rho_v2_2),
      device_element.get_own(rho_e_2),
      device_element.get_own(rho_next),
      device_element.get_own(rho_v1_next),
      device_element.get_own(rho_v2_next),
      device_element.get_own(rho_e_next),
      device_element.get_own(volume),
      device_element.get_own(rho_fluxes),
      device_element.get_own(rho_v1_fluxes),
      device_element.get_own(rho_v2_fluxes),
      device_element.get_own(rho_e_fluxes),
      delta_t, num_local_elements);
  T8GPU_CUDA_CHECK_LAST_ERROR();
}

__global__ void estimate_gradient(float_type const* const* __restrict__ rho,
				  float_type** __restrict__ rho_gradient,
				  float_type const* __restrict__ normal,
				  float_type const* __restrict__ area,
				  int const* e_idx, int* rank,
				  t8_locidx_t* indices, int nb_edges) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_edges) return;

  int l_rank  = rank[e_idx[2 * i]];
  int l_index = indices[e_idx[2 * i]];

  int r_rank  = rank[e_idx[2 * i + 1]];
  int r_index = indices[e_idx[2 * i + 1]];

  float_type rho_l    = rho[l_rank][l_index];
  float_type rho_r    = rho[r_rank][r_index];

  float_type gradient = abs(rho_r - rho_l);

  atomicAdd(&rho_gradient[l_rank][l_index], gradient);
  atomicAdd(&rho_gradient[r_rank][r_index], gradient);
}

void t8gpu::AdvectionSolver::adapt() {
  constexpr int thread_block_size = 256;
  const int gradient_num_blocks = (num_local_faces + thread_block_size - 1) / thread_block_size;
  estimate_gradient<<<gradient_num_blocks, thread_block_size>>>(
	device_element.get_all(rho_next),
	device_element.get_all(rho_fluxes),
	thrust::raw_pointer_cast(device_face_normals.data()),
	thrust::raw_pointer_cast(device_face_area.data()),
	thrust::raw_pointer_cast(device_face_neighbors.data()),
	thrust::raw_pointer_cast(device_ranks.data()),
	thrust::raw_pointer_cast(device_indices.data()),
	num_local_faces);
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(comm);

  const int fluxes_num_blocks = (num_local_elements + thread_block_size - 1) / thread_block_size;
  compute_refinement_criteria<<<fluxes_num_blocks, thread_block_size>>>(
	device_element.get_own(rho_fluxes),
	device_element.get_own(volume),
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

  thrust::host_vector<float_type> adapted_element_variable(num_new_elements);
  thrust::host_vector<float_type> adapted_element_volume(num_new_elements);
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

  forest_user_data_t* forest_user_data {static_cast<forest_user_data_t*>(t8_forest_get_user_data(forest))};
  assert(forest_user_data != nullptr);

  t8_forest_set_user_data(adapted_forest, forest_user_data);
  t8_forest_unref(&forest);

  thrust::device_vector<float_type> device_element_rho_next_adapted(num_new_elements);
  thrust::device_vector<float_type> device_element_rho_v1_next_adapted(num_new_elements);
  thrust::device_vector<float_type> device_element_rho_v2_next_adapted(num_new_elements);
  thrust::device_vector<float_type> device_element_rho_e_next_adapted(num_new_elements);

  thrust::device_vector<float_type> device_element_volume_adapted(num_new_elements);
  t8_locidx_t* device_element_adapt_data {};
  T8GPU_CUDA_CHECK_ERROR(cudaMalloc(&device_element_adapt_data, (num_new_elements + 1) * sizeof(t8_locidx_t)));
  T8GPU_CUDA_CHECK_ERROR(
      cudaMemcpy(device_element_adapt_data, element_adapt_data.data(), element_adapt_data.size() * sizeof(t8_locidx_t), cudaMemcpyHostToDevice));
  const int adapt_num_blocks = (num_new_elements + thread_block_size - 1) / thread_block_size;
  adapt_variables_and_volume<<<adapt_num_blocks, thread_block_size>>>(
      device_element.get_own(rho_next),
      device_element.get_own(rho_v1_next),
      device_element.get_own(rho_v2_next),
      device_element.get_own(rho_e_next),
      device_element.get_own(volume),
      thrust::raw_pointer_cast(device_element_rho_next_adapted.data()),
      thrust::raw_pointer_cast(device_element_rho_v1_next_adapted.data()),
      thrust::raw_pointer_cast(device_element_rho_v2_next_adapted.data()),
      thrust::raw_pointer_cast(device_element_rho_e_next_adapted.data()),
      thrust::raw_pointer_cast(device_element_volume_adapted.data()),
      device_element_adapt_data, num_new_elements);
  T8GPU_CUDA_CHECK_LAST_ERROR();
  T8GPU_CUDA_CHECK_ERROR(cudaFree(device_element_adapt_data));

  // resize shared and owned element variables
  device_element.resize(num_new_elements);

  element_refinement_criteria.resize(num_new_elements);
  device_element_refinement_criteria.resize(num_new_elements);

  // fill fluxes device element variable
  float_type* device_element_rho_fluxes_ptr {device_element.get_own(rho_fluxes)};
  T8GPU_CUDA_CHECK_ERROR(cudaMemset(device_element_rho_fluxes_ptr, 0, 4*sizeof(float_type)*num_new_elements));

  // TODO add copy with rvalue reference
  device_element.copy(rho_next, std::move(device_element_rho_next_adapted));
  device_element.copy(rho_v1_next, std::move(device_element_rho_v1_next_adapted));
  device_element.copy(rho_v2_next, std::move(device_element_rho_v2_next_adapted));
  device_element.copy(rho_e_next, std::move(device_element_rho_e_next_adapted));
  device_element.copy(volume, std::move(device_element_volume_adapted));


  forest = adapted_forest;

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

  thrust::device_vector<float_type> device_new_element_rho(num_new_elements);
  thrust::device_vector<float_type> device_new_element_rho_v1(num_new_elements);
  thrust::device_vector<float_type> device_new_element_rho_v2(num_new_elements);
  thrust::device_vector<float_type> device_new_element_rho_e(num_new_elements);
  thrust::device_vector<float_type> device_new_element_volume(num_new_elements);

  constexpr int thread_block_size = 256;
  const int fluxes_num_blocks = (num_new_elements + thread_block_size - 1) / thread_block_size;
  partition_data<<<fluxes_num_blocks, thread_block_size>>>(thrust::raw_pointer_cast(device_new_ranks.data()),
							   thrust::raw_pointer_cast(device_new_indices.data()),
							   thrust::raw_pointer_cast(device_new_element_rho.data()),
							   thrust::raw_pointer_cast(device_new_element_rho_v1.data()),
							   thrust::raw_pointer_cast(device_new_element_rho_v2.data()),
							   thrust::raw_pointer_cast(device_new_element_rho_e.data()),
							   thrust::raw_pointer_cast(device_new_element_volume.data()),
							   device_element.get_all(rho_next),
							   device_element.get_all(rho_v1_next),
							   device_element.get_all(rho_v2_next),
							   device_element.get_all(rho_e_next),
							   device_element.get_all(volume),
							   num_new_elements);
  cudaDeviceSynchronize();
  MPI_Barrier(comm);

  // resize shared and own element variables
  device_element.resize(num_new_elements);

  device_element_refinement_criteria.resize(num_new_elements);

  // copy new shared element variables
  device_element.copy(rho_next, std::move(device_new_element_rho));
  device_element.copy(rho_v1_next, std::move(device_new_element_rho_v1));
  device_element.copy(rho_v2_next, std::move(device_new_element_rho_v2));
  device_element.copy(rho_e_next, std::move(device_new_element_rho_e));
  device_element.copy(volume, std::move(device_new_element_volume));

  float_type* device_element_rho_fluxes_ptr {device_element.get_own(rho_fluxes)};
  T8GPU_CUDA_CHECK_ERROR(cudaMemset(device_element_rho_fluxes_ptr, 0, 4*sizeof(float_type)*num_new_elements));

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
  device_face_speed_estimate.resize(face_area.size());
}

void t8gpu::AdvectionSolver::save_vtk(const std::string& prefix) const {
  save_vtk_impl(prefix);
}

template<typename ft>
void t8gpu::AdvectionSolver::save_vtk_impl(const std::string& prefix) const {
  thrust::host_vector<ft> element_variable(num_local_elements);
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(element_variable.data(), device_element.get_own(rho_next), sizeof(ft)*num_local_elements, cudaMemcpyDeviceToHost));

  t8_vtk_data_field_t vtk_data_field {};
  vtk_data_field.type = T8_VTK_SCALAR;
  strcpy(vtk_data_field.description, "density");
  if constexpr (std::is_same<ft, double>::value) { // no need for conversion
    vtk_data_field.data = element_variable.data();
  t8_forest_write_vtk_ext(forest, prefix.c_str(),
			  true,  /* write_treeid */
			  true,  /* write_mpirank */
			  true,  /* write_level */
			  true,  /* write_element_id */
			  false, /* write_ghost */
			  false, /* write_curved */
			  false, /* do_not_use_API */
			  1,     /* num_data */
			  &vtk_data_field);
  } else { // we need to convert to double precision
    thrust::host_vector<double> double_element_variable(num_local_elements);
    for (t8_locidx_t i=0; i<num_local_elements; i++)
      double_element_variable[i] = static_cast<double>(element_variable[i]);
    vtk_data_field.data = double_element_variable.data();
  t8_forest_write_vtk_ext(forest, prefix.c_str(),
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
}

float_type t8gpu::AdvectionSolver::compute_integral() const {
  float_type local_integral = 0.0;
  float_type const* mem {device_element.get_own(rho_next)};
  thrust::host_vector<float_type> variable(num_local_elements);
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(variable.data(), mem, sizeof(float_type)*device_element.size(), cudaMemcpyDeviceToHost));
  thrust::host_vector<float_type> volume(num_local_elements);
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(volume.data(), device_element.get_own(VariableName::volume), sizeof(float_type)*device_element.size(), cudaMemcpyDeviceToHost));

  for (t8_locidx_t i=0; i<num_local_elements; i++) {
    local_integral += volume[i] * variable[i];
  }
  float_type global_integral {};
  if constexpr (std::is_same<float_type, double>::value) {
    MPI_Allreduce(&local_integral, &global_integral, 1, MPI_DOUBLE, MPI_SUM, comm);
  } else {
    MPI_Allreduce(&local_integral, &global_integral, 1, MPI_FLOAT, MPI_SUM, comm);
  }
  return global_integral;
}

float_type t8gpu::AdvectionSolver::compute_timestep() const {
  float_type local_speed_estimate = thrust::reduce(device_face_speed_estimate.begin(),
						   device_face_speed_estimate.end(),
						   float_type{0.0}, thrust::maximum<float_type>());
  float_type global_speed_estimate {};
  if constexpr (std::is_same<float_type, double>::value) {
    MPI_Allreduce(&local_speed_estimate, &global_speed_estimate, 1, MPI_DOUBLE, MPI_MAX, comm);
  } else {
    MPI_Allreduce(&local_speed_estimate, &global_speed_estimate, 1, MPI_FLOAT, MPI_MAX, comm);
  }

  return  cfl*static_cast<float_type>(std::pow(static_cast<float_type>(0.5), max_level))/global_speed_estimate;
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
	    face_normals.push_back(static_cast<float_type>(face_normal[0]));
	    face_normals.push_back(static_cast<float_type>(face_normal[1]));
	    face_area.push_back(static_cast<float_type>(t8_forest_element_face_area(forest, tree_idx, element, face_idx)) / static_cast<float_type>(num_neighbors));
	  }
	}

        if ((num_neighbors == 1) && (neighbor_ids[0] < num_local_elements) &&
            ((neighbor_ids[0] > element_idx) ||
             (neighbor_ids[0] < element_idx && neigh_scheme[0].t8_element_level(neighbors[0]) < eclass_scheme->t8_element_level(element)))) {
	  face_neighbors.push_back(element_idx);
	  face_neighbors.push_back(neighbor_ids[0]);
	  double face_normal[3];
	  t8_forest_element_face_normal(forest, tree_idx, element, face_idx, face_normal);
	  face_normals.push_back(static_cast<float_type>(face_normal[0]));
	  face_normals.push_back(static_cast<float_type>(face_normal[1]));
	  face_area.push_back(static_cast<float_type>(t8_forest_element_face_area(forest, tree_idx, element, face_idx)));
        }
	neigh_scheme->t8_element_destroy(num_neighbors, neighbors);
        T8_FREE(neighbors);

        T8_FREE(dual_faces);
        T8_FREE(neighbor_ids);
      }

      element_idx++;
    }
  }

  num_local_faces = static_cast<t8_locidx_t>(face_area.size());
}

void t8gpu::AdvectionSolver::compute_fluxes(VariableName rho,
					    VariableName rho_v1,
					    VariableName rho_v2,
					    VariableName rho_e) {
  constexpr int thread_block_size = 256;
  const int fluxes_num_blocks = (num_local_faces + thread_block_size - 1) / thread_block_size;
  kepes_compute_fluxes<<<fluxes_num_blocks, thread_block_size>>>(
								 device_element.get_all(rho),
								 device_element.get_all(rho_v1),
								 device_element.get_all(rho_v2),
								 device_element.get_all(rho_e),
								 device_element.get_all(rho_fluxes),
								 device_element.get_all(rho_v1_fluxes),
								 device_element.get_all(rho_v2_fluxes),
								 device_element.get_all(rho_e_fluxes),
								 thrust::raw_pointer_cast(device_face_speed_estimate.data()),
								 thrust::raw_pointer_cast(device_face_normals.data()),
								 thrust::raw_pointer_cast(device_face_area.data()),
								 thrust::raw_pointer_cast(device_face_neighbors.data()),
								 thrust::raw_pointer_cast(device_ranks.data()),
								 thrust::raw_pointer_cast(device_indices.data()),
								 num_local_faces);
  T8GPU_CUDA_CHECK_LAST_ERROR();
  cudaDeviceSynchronize();
  MPI_Barrier(comm);
}

static int adapt_callback_initialization(t8_forest_t forest, t8_forest_t forest_from, t8_locidx_t which_tree, t8_locidx_t lelement_id,
					 t8_eclass_scheme_c* ts, const int is_family, const int num_elements, t8_element_t* elements[]) {
  // t8_locidx_t element_level {ts->t8_element_level(elements[0])};

  // double b = 0.02;

  // if (element_level < t8gpu::AdvectionSolver::max_level) {
  //   double center[3];
  //   t8_forest_element_centroid(forest_from, which_tree, elements[0], center);

  //   double variable = sqrt((0.5 - center[0]) * (0.5 - center[0]) + (0.5 - center[1]) * (0.5 - center[1])) - 0.25;

  //   if (std::abs(variable) < b) return 1;
  // }
  // if (element_level > t8gpu::AdvectionSolver::min_level && is_family) {
  //   double center[] = {0.0, 0.0, 0.0};
  //   double current_element_center[] = {0.0, 0.0, 0.0};
  //   for (size_t i = 0; i < 4; i++) {
  //     t8_forest_element_centroid(forest_from, which_tree, elements[i], current_element_center);
  //     for (size_t j = 0; j < 3; j++) {
  //       center[j] += current_element_center[j] / 4.0;
  //     }
  //   }

  //   double variable = sqrt((0.5 - center[0]) * (0.5 - center[0]) + (0.5 - center[1]) * (0.5 - center[1])) - 0.25;

  //   if (std::abs(variable) > b) return -1;
  // }

  return 0;
}

static int adapt_callback_iteration(t8_forest_t forest, t8_forest_t forest_from, t8_locidx_t which_tree, t8_locidx_t lelement_id, t8_eclass_scheme_c* ts,
				    const int is_family, const int num_elements, t8_element_t* elements[]) {
  forest_user_data_t* forest_user_data = static_cast<forest_user_data_t*>(t8_forest_get_user_data(forest_from));
  assert(forest_user_data != nullptr);

  t8_locidx_t element_level {ts->t8_element_level(elements[0])};

  t8_locidx_t tree_offset = t8_forest_get_tree_element_offset(forest_from, which_tree);

  float_type b = static_cast<float_type>(10.0);

  if (element_level < t8gpu::AdvectionSolver::max_level) {
    float_type criteria = (*forest_user_data->element_refinement_criteria)[tree_offset + lelement_id];

    if (criteria > b) {
      return 1;
    }
  }
  if (element_level > t8gpu::AdvectionSolver::min_level && is_family) {
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

__global__ static void compute_refinement_criteria(float_type const* __restrict__ fluxes_rho,
						   float_type const* __restrict__ volume,
						   float_type* __restrict__ criteria, int nb_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_elements) return;

  criteria[i] = fluxes_rho[i] / sqrt(volume[i]);
}

__global__ static void adapt_variables_and_volume(float_type const* __restrict__ rho_old,
						  float_type const* __restrict__ rho_v1_old,
						  float_type const* __restrict__ rho_v2_old,
						  float_type const* __restrict__ rho_e_old,
						  float_type const* __restrict__ volume_old,
						  float_type* __restrict__ rho_new,
						  float_type* __restrict__ rho_v1_new,
						  float_type* __restrict__ rho_v2_new,
						  float_type* __restrict__ rho_e_new,
						  float_type* __restrict__ volume_new,
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

  rho_new[i] = 0.0;
  rho_v1_new[i] = 0.0;
  rho_v2_new[i] = 0.0;
  rho_e_new[i] = 0.0;
  for (int j = 0; j < nb_elements_sum; j++) {
    rho_new[i]    += rho_old[adapt_data[i] + j] / static_cast<float_type>(nb_elements_sum);
    rho_v1_new[i] += rho_v1_old[adapt_data[i] + j] / static_cast<float_type>(nb_elements_sum);
    rho_v2_new[i] += rho_v2_old[adapt_data[i] + j] / static_cast<float_type>(nb_elements_sum);
    rho_e_new[i]  += rho_e_old[adapt_data[i] + j] / static_cast<float_type>(nb_elements_sum);
  }
}

__global__ void partition_data(int* __restrict__ ranks, t8_locidx_t* __restrict__ indices,
			       float_type* __restrict__ new_rho,
			       float_type* __restrict__ new_rho_v1,
			       float_type* __restrict__ new_rho_v2,
			       float_type* __restrict__ new_rho_e,
			       float_type* __restrict__ new_volume,
			       float_type const*const* __restrict__ old_rho,
			       float_type const*const* __restrict__ old_rho_v1,
			       float_type const*const* __restrict__ old_rho_v2,
			       float_type const*const* __restrict__ old_rho_e,
			       float_type const*const* __restrict__ old_volume,
			       int num_new_elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_new_elements) return;

  new_rho[i]    = old_rho[ranks[i]][indices[i]];
  new_rho_v1[i] = old_rho_v1[ranks[i]][indices[i]];
  new_rho_v2[i] = old_rho_v2[ranks[i]][indices[i]];
  new_rho_e[i]  = old_rho_e[ranks[i]][indices[i]];

  new_volume[i] = old_volume[ranks[i]][indices[i]];
}

template<typename ft>
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

using nc = numerical_constants<float_type>;

__global__ static void hll_compute_fluxes(float_type** __restrict__ rho,
					  float_type** __restrict__ rho_v1,
					  float_type** __restrict__ rho_v2,
					  float_type** __restrict__ rho_e,
					  float_type** __restrict__ rho_fluxes,
					  float_type** __restrict__ rho_v1_fluxes,
					  float_type** __restrict__ rho_v2_fluxes,
					  float_type** __restrict__ rho_e_fluxes,
					  float_type* __restrict__ speed_estimate,
					  float_type const* __restrict__ normal,
					  float_type const* __restrict__ area,
					  int const* e_idx, int* rank,
					  t8_locidx_t* indices, int nb_edges) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_edges) return;

  float_type gamma = t8gpu::AdvectionSolver::gamma;

  float_type face_surface = area[i];

  int l_rank  = rank[e_idx[2 * i]];
  int l_index = indices[e_idx[2 * i]];

  int r_rank  = rank[e_idx[2 * i + 1]];
  int r_index = indices[e_idx[2 * i + 1]];

  float_type nx = normal[2*i];
  float_type ny = normal[2*i+1];

  float_type rho_l    = rho[l_rank][l_index];
  float_type rho_vx_l = rho_v1[l_rank][l_index];
  float_type rho_vy_l = rho_v2[l_rank][l_index];
  float_type rho_e_l  = rho_e[l_rank][l_index];

  float_type rho_r    = rho[r_rank][r_index];
  float_type rho_vx_r = rho_v1[r_rank][r_index];
  float_type rho_vy_r = rho_v2[r_rank][r_index];
  float_type rho_e_r  = rho_e[r_rank][r_index];

  // rotate from (x,y) basis to local basis (n,t)
  float_type rho_v1_l =  nx*rho_vx_l + ny*rho_vy_l;
  float_type rho_v2_l = -ny*rho_vx_l + nx*rho_vy_l;

  float_type rho_v1_r =  nx*rho_vx_r + ny*rho_vy_r;
  float_type rho_v2_r = -ny*rho_vx_r + nx*rho_vy_r;

  float_type v1_l = rho_v1_l/rho_l;
  float_type v2_l = rho_v2_l/rho_l;
  float_type p_l = (gamma-1)*(rho_e_l-nc::half*rho_l*(v1_l*v1_l+v2_l*v2_l));
  float_type H_l = (rho_e_l+p_l)/rho_l;
  float_type c_l = sqrt((gamma-1)*(H_l-nc::half*(v1_l*v1_l+v2_l*v2_l)));

  float_type v1_r = rho_v1_r/rho_r;
  float_type v2_r = rho_v2_r/rho_r;
  float_type p_r = (gamma-nc::one)*(rho_e_r-nc::half*rho_r*(v1_r*v1_r+v2_r*v2_r));
  float_type H_r = (rho_e_r+p_r)/rho_r;
  float_type c_r = sqrt((gamma-nc::one)*(H_r-nc::half*(v1_r*v1_r+v2_r*v2_r)));

  float_type sqrt_rho_l = sqrt(rho_l);
  float_type sqrt_rho_r = sqrt(rho_r);
  float_type sum_weights = sqrt_rho_l+sqrt_rho_r;

  float_type v1_roe = (sqrt_rho_l*v1_l+sqrt_rho_r*v1_r)/sum_weights;
  float_type v2_roe = (sqrt_rho_l*v2_l+sqrt_rho_r*v2_r)/sum_weights;
  float_type H_roe = (sqrt_rho_l*H_l+sqrt_rho_r*H_r)/sum_weights;
  float_type c_roe = sqrt((gamma-nc::one)*(H_roe-nc::half*(v1_roe*v1_roe+v2_roe*v2_roe)));

  float_type S_l = min(v1_roe-c_roe, v1_l-c_l);
  float_type S_r = max(v1_roe+c_roe, v1_r+c_r);

  speed_estimate[i] = max(-S_l, S_r);

  float_type F_l[4] = {rho_v1_l,
    rho_v1_l*rho_v1_l/rho_l + p_l,
    rho_v1_l*v2_l,
    rho_v1_l*H_l};

  float_type F_r[4] = {rho_v1_r,
    rho_v1_r*rho_v1_r/rho_r + p_r,
    rho_v1_r*v2_r,
    rho_v1_r*H_r};

  float_type S_l_clamp = min(S_l, nc::zero);
  float_type S_r_clamp = max(S_r, nc::zero);

  float_type rho_flux    = face_surface*((S_r_clamp*F_l[0]-S_l_clamp*F_r[0])+S_r_clamp*S_l_clamp*(rho_r-rho_l))/(S_r_clamp-S_l_clamp);
  float_type rho_v1_flux = face_surface*((S_r_clamp*F_l[1]-S_l_clamp*F_r[1])+S_r_clamp*S_l_clamp*(rho_v1_r-rho_v1_l))/(S_r_clamp-S_l_clamp);
  float_type rho_v2_flux = face_surface*((S_r_clamp*F_l[2]-S_l_clamp*F_r[2])+S_r_clamp*S_l_clamp*(rho_v2_r-rho_v2_l))/(S_r_clamp-S_l_clamp);
  float_type rho_e_flux  = face_surface*((S_r_clamp*F_l[3]-S_l_clamp*F_r[3])+S_r_clamp*S_l_clamp*(rho_e_r-rho_e_l))/(S_r_clamp-S_l_clamp);

  // rotate back
  float_type rho_vx_flux = nx*rho_v1_flux - ny*rho_v2_flux;
  float_type rho_vy_flux = ny*rho_v1_flux + nx*rho_v2_flux;

  atomicAdd(&rho_fluxes[l_rank][l_index], -rho_flux);
  atomicAdd(&rho_fluxes[r_rank][r_index],  rho_flux);

  atomicAdd(&rho_v1_fluxes[l_rank][l_index], -rho_vx_flux);
  atomicAdd(&rho_v1_fluxes[r_rank][r_index],  rho_vx_flux);

  atomicAdd(&rho_v2_fluxes[l_rank][l_index], -rho_vy_flux);
  atomicAdd(&rho_v2_fluxes[r_rank][r_index],  rho_vy_flux);

  atomicAdd(&rho_e_fluxes[l_rank][l_index], -rho_e_flux);
  atomicAdd(&rho_e_fluxes[r_rank][r_index],  rho_e_flux);
}

__device__ static float ln_mean(float aL, float aR) {
  float Xi = aR/aL;
  float u = (Xi*(Xi-2.0f)+1.0f)/(Xi*(Xi+2.0f)+1.0f);

  float eps = 1.0e-4f;
  if (u < eps) {
    return (aL+aR)*52.50f/(105.0f + u*(35.0f + u*(21.0f + u*15.0f)));
  } else {
    return (aR-aL)/logf(Xi);
  }
}

__device__ static double ln_mean(double aL, double aR) {
  double Xi = aR/aL;
  double u = (Xi*(Xi-2.0)+1.0)/(Xi*(Xi+2.0)+1.0);

  double eps = 1.0e-4;
  if (u < eps) {
    return (aL+aR)*52.50/(105.0 + u*(35.0 + u*(21.0 + u*15.0)));
  } else {
    return (aR-aL)/log(Xi);
  }
}

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
  float_type kappa = t8gpu::AdvectionSolver::gamma;
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
  rhoHat    = ln_mean(u_L[0],u_R[0]);
  float_type beta_MEAN = nc::half*(beta_L+beta_R);
  float_type beta_Hat  = ln_mean(beta_L,beta_R);

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



  float_type kappa = t8gpu::AdvectionSolver::gamma;
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

__global__ static void kepes_compute_fluxes(float_type** __restrict__ rho,
					    float_type** __restrict__ rho_v1,
					    float_type** __restrict__ rho_v2,
					    float_type** __restrict__ rho_e,
					    float_type** __restrict__ rho_fluxes,
					    float_type** __restrict__ rho_v1_fluxes,
					    float_type** __restrict__ rho_v2_fluxes,
					    float_type** __restrict__ rho_e_fluxes,
					    float_type* __restrict__ speed_estimate,
					    float_type const* __restrict__ normal,
					    float_type const* __restrict__ area,
					    int const* e_idx, int* rank,
					    t8_locidx_t* indices, int nb_edges) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_edges) return;

  float_type face_surface = area[i];

  int l_rank  = rank[e_idx[2 * i]];
  int l_index = indices[e_idx[2 * i]];

  int r_rank  = rank[e_idx[2 * i + 1]];
  int r_index = indices[e_idx[2 * i + 1]];

  float_type nx = normal[2*i];
  float_type ny = normal[2*i+1];

  float_type rho_l    = rho[l_rank][l_index];
  float_type rho_vx_l = rho_v1[l_rank][l_index];
  float_type rho_vy_l = rho_v2[l_rank][l_index];
  float_type rho_e_l  = rho_e[l_rank][l_index];

  float_type rho_r    = rho[r_rank][r_index];
  float_type rho_vx_r = rho_v1[r_rank][r_index];
  float_type rho_vy_r = rho_v2[r_rank][r_index];
  float_type rho_e_r  = rho_e[r_rank][r_index];

  // rotate from (x,y) basis to local basis (n,t)
  float_type rho_v1_l =  nx*rho_vx_l + ny*rho_vy_l;
  float_type rho_v2_l = -ny*rho_vx_l + nx*rho_vy_l;

  float_type rho_v1_r =  nx*rho_vx_r + ny*rho_vy_r;
  float_type rho_v2_r = -ny*rho_vx_r + nx*rho_vy_r;

  float_type u_L[5] = {rho_l, rho_v1_l, rho_v2_l, nc::zero, rho_e_l};
  float_type u_R[5] = {rho_r, rho_v1_r, rho_v2_r, nc::zero, rho_e_r};

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

  speed_estimate[i] = abs(uHat) + aHat;


  float_type kappa = t8gpu::AdvectionSolver::gamma;
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
  float_type rho_e_flux  = face_surface*F[4];

  // rotate back
  float_type rho_vx_flux = nx*rho_v1_flux - ny*rho_v2_flux;
  float_type rho_vy_flux = ny*rho_v1_flux + nx*rho_v2_flux;

  atomicAdd(&rho_fluxes[l_rank][l_index], -rho_flux);
  atomicAdd(&rho_fluxes[r_rank][r_index],  rho_flux);

  atomicAdd(&rho_v1_fluxes[l_rank][l_index], -rho_vx_flux);
  atomicAdd(&rho_v1_fluxes[r_rank][r_index],  rho_vx_flux);

  atomicAdd(&rho_v2_fluxes[l_rank][l_index], -rho_vy_flux);
  atomicAdd(&rho_v2_fluxes[r_rank][r_index],  rho_vy_flux);

  atomicAdd(&rho_e_fluxes[l_rank][l_index], -rho_e_flux);
  atomicAdd(&rho_e_fluxes[r_rank][r_index],  rho_e_flux);
}
