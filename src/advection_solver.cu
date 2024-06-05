#include <advection_solver.h>
#include <cuda_utils.h>

#include <cassert>
#include <cmath>
#include <iostream>

#include <t8.h>
#include <t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_examples.h>
#include <t8_forest/t8_forest.h>
#include <t8_forest/t8_forest_iterate.h>
#include <t8_schemes/t8_default/t8_default_cxx.hxx>

struct forest_user_data_t {
  std::vector<double>* element_variable;
  std::vector<double>* element_volume;
};

int adapt_callback_initialization(t8_forest_t forest, t8_forest_t forest_from, t8_locidx_t which_tree, t8_locidx_t lelement_id,
				  t8_eclass_scheme_c* ts, const int is_family, const int num_elements, t8_element_t* elements[]) {

  t8_locidx_t element_level = ts->t8_element_level(elements[0]);

  double b = 0.02;

  if (element_level < advection_solver_t::max_level) {
    double center[3];
    t8_forest_element_centroid(forest_from, which_tree, elements[0], center);

    double variable = sqrt((0.5-center[0])*(0.5-center[0]) + (0.5-center[1])*(0.5-center[1])) - 0.25;

    if (std::abs(variable) < b)
      return 1;
  }
  if (element_level > advection_solver_t::min_level && is_family) {
    double center[] = {0.0, 0.0, 0.0};
    double current_element_center[] = {0.0, 0.0, 0.0};
    for (size_t i=0; i<4; i++) {
      t8_forest_element_centroid(forest_from, which_tree, elements[i], current_element_center);
      for (size_t j=0; j<3; j++) {
	center[j] += current_element_center[j] / 4.0;
      }
    }

    double variable = sqrt((0.5-center[0])*(0.5-center[0]) + (0.5-center[1])*(0.5-center[1])) - 0.25;

    if (std::abs(variable) > b)
      return -1;
  }

  return 0;
}

int adapt_callback_iteration(t8_forest_t forest, t8_forest_t forest_from, t8_locidx_t which_tree, t8_locidx_t lelement_id,
			     t8_eclass_scheme_c* ts, const int is_family, const int num_elements, t8_element_t* elements[]) {
  forest_user_data_t* forest_user_data = static_cast<forest_user_data_t*>(t8_forest_get_user_data(forest_from));

  t8_locidx_t element_level = ts->t8_element_level(elements[0]);

  double b = 1.0;
  double h = std::pow(0.5, element_level);

  if (element_level < advection_solver_t::max_level) {
    double center[3];
    t8_forest_element_centroid(forest_from, which_tree, elements[0], center);

    double variable = (*forest_user_data->element_variable)[lelement_id];

    if (std::abs(variable) < b*h) {
      return 1;
    }
  }
  if (element_level > advection_solver_t::min_level && is_family) {
    double variable = 0.0;
    for (size_t i=0; i<4; i++) {
      variable += (*forest_user_data->element_variable)[lelement_id + i] / 4.0;
    }

    if (std::abs(variable) > (2*h)*b) {
      return -1;
    }
  }

  return 0;
}


void iterate_replace_callback(t8_forest_t forest_old, t8_forest_t forest_new, t8_locidx_t which_tree,
			      t8_eclass_scheme_c* ts, const int refine, const int num_outgoing,
			      t8_locidx_t first_outgoing, const int num_incoming,
			      t8_locidx_t first_incoming) {

  forest_user_data_t* forest_user_data_new = static_cast<forest_user_data_t*>(t8_forest_get_user_data(forest_new));
  forest_user_data_t* forest_user_data_old = static_cast<forest_user_data_t*>(t8_forest_get_user_data(forest_old));

  first_incoming += t8_forest_get_tree_element_offset(forest_new, which_tree);
  first_outgoing += t8_forest_get_tree_element_offset(forest_old, which_tree);

  /* Do not adapt or coarsen */
  if (refine == 0) {
    (*forest_user_data_new->element_variable)[first_incoming] = (*forest_user_data_old->element_variable)[first_outgoing];
    (*forest_user_data_new->element_volume)[first_incoming] = (*forest_user_data_old->element_volume)[first_outgoing];
  }
  /* The old element is refined, we copy the element values */
  else if (refine == 1) {
    for (int i = 0; i < num_incoming; i++) {
      (*forest_user_data_new->element_variable)[first_incoming + i] = (*forest_user_data_old->element_variable)[first_outgoing];
      (*forest_user_data_new->element_volume)[first_incoming + i] = (*forest_user_data_old->element_volume)[first_outgoing] / 4.0;
    }
  }
  /* Old element is coarsened */
  else if (refine == -1) {
    double tmp_variable = 0.0;
    for (t8_locidx_t i = 0; i < num_outgoing; i++) {
      tmp_variable += (*forest_user_data_old->element_variable)[first_outgoing + i] / 4.0;
    }
    (*forest_user_data_new->element_variable)[first_incoming] = tmp_variable;
    (*forest_user_data_new->element_volume)[first_incoming] = (*forest_user_data_old->element_volume)[first_outgoing] * 4.0;
  }
}

advection_solver_t::advection_solver_t() : comm(sc_MPI_COMM_WORLD),
					   cmesh(t8_cmesh_new_periodic(comm, dim)),
					   scheme(t8_scheme_new_default_cxx()),
					   forest(t8_forest_new_uniform(cmesh, scheme, 6, false, comm)),
					   element_variable(t8_forest_get_local_num_elements(forest)),
					   element_volume(t8_forest_get_local_num_elements(forest)),
					   delta_t(1.0*std::pow(0.5, max_level) / sqrt(2.0)) {

  t8_forest_t new_forest;
  t8_forest_init(&new_forest);
  t8_forest_set_adapt(new_forest, forest, adapt_callback_initialization, true);
  t8_forest_set_balance(new_forest, forest, true);
  t8_forest_commit(new_forest);
  forest = new_forest;

  element_variable.resize(t8_forest_get_local_num_elements(forest));
  element_volume.resize(t8_forest_get_local_num_elements(forest));

  t8_locidx_t num_local_elements = element_variable.size();

  t8_locidx_t num_local_trees = t8_forest_get_num_local_trees(forest);
  t8_locidx_t element_idx = 0;
  for (t8_locidx_t tree_idx = 0; tree_idx < num_local_trees; tree_idx++) {
    t8_eclass_t tree_class = t8_forest_get_tree_class (forest, tree_idx);
    t8_eclass_scheme_c *eclass_scheme = t8_forest_get_eclass_scheme (forest, tree_class);

    t8_locidx_t num_elements_in_tree = t8_forest_get_tree_num_elements(forest, tree_idx);
    for (t8_locidx_t tree_element_idx = 0; tree_element_idx < num_elements_in_tree; tree_element_idx++) {
      const t8_element_t* element = t8_forest_get_element_in_tree(forest, tree_idx, tree_element_idx);

      double center[3];
      t8_forest_element_centroid(forest, tree_idx, element, center);

      element_variable[element_idx] = sqrt((0.5-center[0])*(0.5-center[0]) + (0.5-center[1])*(0.5-center[1])) - 0.25;
      element_volume[element_idx] = t8_forest_element_volume(forest, tree_idx, element);

      size_t num_faces = static_cast<size_t>(eclass_scheme->t8_element_num_faces(element));
      for (size_t face_idx=0; face_idx < num_faces; face_idx++) {
	int num_neighbors;
	int* dual_faces;
	t8_locidx_t* neighbor_ids;
	t8_element_t** neighbors;
	t8_eclass_scheme_c* neigh_scheme;

	t8_forest_leaf_face_neighbors(forest, tree_idx, element, &neighbors, face_idx, &dual_faces, &num_neighbors,
				      &neighbor_ids, &neigh_scheme, 1);

	if ((num_neighbors == 1) && ((neighbor_ids[0] > element_idx) ||
				     (neighbor_ids[0] < element_idx  && neigh_scheme[0].t8_element_level(neighbors[0]) < eclass_scheme->t8_element_level(element)
				      ))) {
	  face_neighbors.push_back(std::array<t8_locidx_t,2>{element_idx, neighbor_ids[0]});
	  double face_normal[3];
	  t8_forest_element_face_normal(forest, tree_idx, element, face_idx, face_normal);
	  face_normals.push_back(std::array<double,2>{face_normal[0], face_normal[1]});
	  face_area.push_back(t8_forest_element_face_area(forest, tree_idx, element, face_idx));
	}

	T8_FREE(neighbors);
	T8_FREE(dual_faces);
	T8_FREE(neighbor_ids);
      }

      element_idx++;
    }
  }

  CUDA_CHECK_ERROR(cudaMalloc(&device_element_variable_prev, sizeof(double)*num_local_elements));
  CUDA_CHECK_ERROR(cudaMalloc(&device_element_variable_next, sizeof(double)*num_local_elements));
  CUDA_CHECK_ERROR(cudaMalloc(&device_element_fluxes, sizeof(double)*num_local_elements));
  CUDA_CHECK_ERROR(cudaMalloc(&device_element_volume, sizeof(double)*num_local_elements));

  CUDA_CHECK_ERROR(cudaMalloc(&device_face_neighbors, sizeof(int)*face_neighbors.size()*2));
  CUDA_CHECK_ERROR(cudaMalloc(&device_face_normals, sizeof(double)*face_normals.size()*2));
  CUDA_CHECK_ERROR(cudaMalloc(&device_face_area, sizeof(double)*face_normals.size()*2));

  CUDA_CHECK_ERROR(cudaMemcpy(device_element_variable_next, element_variable.data(), element_variable.size()*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(device_element_volume, element_volume.data(), element_volume.size()*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemset(device_element_fluxes, 0, element_variable.size()));

  CUDA_CHECK_ERROR(cudaMemcpy(device_face_neighbors, face_neighbors.data(), face_neighbors.size()*2*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(device_face_normals, face_normals.data(), face_normals.size()*2*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(device_face_area, face_area.data(), face_area.size()*2*sizeof(double), cudaMemcpyHostToDevice));

  // TODO: remove allocation out of RAII paradigm
  forest_user_data_t* forest_user_data = static_cast<forest_user_data_t*>(malloc(sizeof(forest_user_data_t)));
  forest_user_data->element_variable = &element_variable;
  forest_user_data->element_volume = &element_volume;
  t8_forest_set_user_data(forest, forest_user_data);
}

void advection_solver_t::adapt() {
  CUDA_CHECK_ERROR(cudaMemcpy(element_variable.data(), device_element_variable_next, element_variable.size()*sizeof(double), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  t8_forest_ref(forest);

  t8_forest_t adapted_forest;
  t8_forest_init(&adapted_forest);
  t8_forest_set_adapt(adapted_forest, forest, adapt_callback_iteration, false);
  t8_forest_set_balance(adapted_forest, forest, true);
  t8_forest_commit(adapted_forest);

  std::vector<double> adapted_element_variable(t8_forest_get_local_num_elements(adapted_forest));
  std::vector<double> adapted_element_volume(t8_forest_get_local_num_elements(adapted_forest));

  forest_user_data_t* adapted_forest_user_data = static_cast<forest_user_data_t*>(malloc(sizeof(forest_user_data_t)));
  adapted_forest_user_data->element_variable = &adapted_element_variable;
  adapted_forest_user_data->element_volume = &adapted_element_volume;
  t8_forest_set_user_data(adapted_forest, adapted_forest_user_data);

  t8_forest_iterate_replace(adapted_forest, forest, iterate_replace_callback);

  element_variable = std::move(adapted_element_variable);
  element_volume = std::move(adapted_element_volume);

  adapted_forest_user_data->element_variable = &element_variable;
  adapted_forest_user_data->element_volume = &element_volume;

  forest_user_data_t* forest_user_data = static_cast<forest_user_data_t*>(t8_forest_get_user_data(forest));
  free(forest_user_data);
  t8_forest_unref(&forest);

  forest = adapted_forest;

  compute_edge_information();

  // TODO: do more clever reallocation
  CUDA_CHECK_ERROR(cudaFree(device_element_variable_prev));
  CUDA_CHECK_ERROR(cudaFree(device_element_variable_next));
  CUDA_CHECK_ERROR(cudaFree(device_element_fluxes));
  CUDA_CHECK_ERROR(cudaFree(device_element_volume));

  CUDA_CHECK_ERROR(cudaFree(device_face_neighbors));
  CUDA_CHECK_ERROR(cudaFree(device_face_normals));
  CUDA_CHECK_ERROR(cudaFree(device_face_area));

  CUDA_CHECK_ERROR(cudaMalloc(&device_element_variable_prev, sizeof(double)*element_variable.size()));
  CUDA_CHECK_ERROR(cudaMalloc(&device_element_variable_next, sizeof(double)*element_variable.size()));
  CUDA_CHECK_ERROR(cudaMalloc(&device_element_fluxes, sizeof(double)*element_variable.size()));
  CUDA_CHECK_ERROR(cudaMalloc(&device_element_volume, sizeof(double)*element_variable.size()));

  CUDA_CHECK_ERROR(cudaMalloc(&device_face_neighbors, sizeof(int)*face_neighbors.size()*2));
  CUDA_CHECK_ERROR(cudaMalloc(&device_face_normals, sizeof(double)*face_normals.size()*2));
  CUDA_CHECK_ERROR(cudaMalloc(&device_face_area, sizeof(double)*face_normals.size()*2));

  CUDA_CHECK_ERROR(cudaMemcpy(device_element_variable_next, element_variable.data(), element_variable.size()*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(device_element_volume, element_volume.data(), element_volume.size()*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemset(device_element_fluxes, 0, element_variable.size()));

  CUDA_CHECK_ERROR(cudaMemcpy(device_face_neighbors, face_neighbors.data(), face_neighbors.size()*2*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(device_face_normals, face_normals.data(), face_normals.size()*2*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(device_face_area, face_area.data(), face_area.size()*2*sizeof(double), cudaMemcpyHostToDevice));
}

advection_solver_t::~advection_solver_t() {
  forest_user_data_t* forest_user_data = static_cast<forest_user_data_t*>(t8_forest_get_user_data(forest));
  free(forest_user_data);

  t8_forest_unref(&forest);
  t8_cmesh_destroy(&cmesh);

  CUDA_CHECK_ERROR(cudaFree(device_element_variable_prev));
  CUDA_CHECK_ERROR(cudaFree(device_element_variable_next));
  CUDA_CHECK_ERROR(cudaFree(device_element_fluxes));
  CUDA_CHECK_ERROR(cudaFree(device_element_volume));

  CUDA_CHECK_ERROR(cudaFree(device_face_neighbors));
  CUDA_CHECK_ERROR(cudaFree(device_face_normals));
  CUDA_CHECK_ERROR(cudaFree(device_face_area));
}

__global__ static void compute_fluxes(double const* __restrict__ variable,
				      double* __restrict__ fluxes,
				      double const* __restrict__ normal,
				      double const* __restrict__ area,
				      int const* e_idx,
				      int nb_edges) {

  double a[2] = {0.5*sqrt(2.0), 0.5*sqrt(2.0)};

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_edges)
    return;

  double flux = area[i]*(a[0]*normal[2*i]+a[1]*normal[2*i+1]);

  if (flux > 0.0) {
    flux *= variable[e_idx[2*i]];
  } else {
    flux *= variable[e_idx[2*i+1]];
  }

  atomicAdd(&fluxes[e_idx[2*i]], -flux);
  atomicAdd(&fluxes[e_idx[2*i+1]], flux);
}

__global__ static void explicit_euler_time_step(double const* __restrict__ variable_prev,
						double* __restrict__ variable_next,
						double const* __restrict__ volume,
						double* __restrict__ fluxes,
						double delta_t,
						int nb_elements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nb_elements)
    return;

  variable_next[i] = variable_prev[i] + delta_t/volume[i]*fluxes[i];

  fluxes[i] = 0.0;
}

void advection_solver_t::iterate() {
  std::swap(device_element_variable_next, device_element_variable_prev);

  constexpr int thread_block_size = 256;
  const int fluxes_num_blocks = face_area.size() / thread_block_size + (face_area.size() % thread_block_size != 0);
  compute_fluxes<<<fluxes_num_blocks, thread_block_size>>>(device_element_variable_prev,
							   device_element_fluxes,
							   device_face_normals,
							   device_face_area,
							   device_face_neighbors,
							   face_neighbors.size());
  CUDA_CHECK_LAST_ERROR();

  const int euler_num_blocks = element_variable.size() / thread_block_size + (element_variable.size() % thread_block_size != 0);
  explicit_euler_time_step<<<euler_num_blocks, thread_block_size>>>(device_element_variable_prev,
								    device_element_variable_next,
								    device_element_volume,
								    device_element_fluxes,
								    delta_t,
								    element_variable.size());
  CUDA_CHECK_LAST_ERROR();
}

void advection_solver_t::save_vtk(const std::string& prefix) {
  CUDA_CHECK_ERROR(cudaMemcpy(element_variable.data(), device_element_variable_next, element_variable.size()*sizeof(double), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  t8_vtk_data_field_t vtk_data_field =  {};
  vtk_data_field.type = T8_VTK_SCALAR;
  strcpy(vtk_data_field.description, "advection variable");
  vtk_data_field.data = element_variable.data();

  t8_forest_write_vtk_ext(forest, prefix.c_str(), 1, 1, 1, 1, 0, 0, 0, 1, &vtk_data_field);
}

void advection_solver_t::compute_edge_information() {
  face_neighbors.clear();
  face_normals.clear();
  face_area.clear();

  t8_locidx_t num_local_elements = t8_forest_get_local_num_elements(forest);

  t8_locidx_t num_local_trees = t8_forest_get_num_local_trees(forest);
  t8_locidx_t element_idx = 0;
  for (t8_locidx_t tree_idx = 0; tree_idx < num_local_trees; tree_idx++) {
    t8_eclass_t tree_class = t8_forest_get_tree_class (forest, tree_idx);
    t8_eclass_scheme_c *eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class);

    t8_locidx_t num_elements_in_tree = t8_forest_get_tree_num_elements(forest, tree_idx);
    for (t8_locidx_t tree_element_idx = 0; tree_element_idx < num_elements_in_tree; tree_element_idx++) {
      const t8_element_t* element = t8_forest_get_element_in_tree(forest, tree_idx, tree_element_idx);

      size_t num_faces = static_cast<size_t>(eclass_scheme->t8_element_num_faces(element));
      for (size_t face_idx=0; face_idx < num_faces; face_idx++) {
	int num_neighbors;
	int* dual_faces;
	t8_locidx_t* neighbor_ids;
	t8_element_t** neighbors;
	t8_eclass_scheme_c* neigh_scheme;

	t8_forest_leaf_face_neighbors(forest, tree_idx, element, &neighbors, face_idx, &dual_faces, &num_neighbors,
				      &neighbor_ids, &neigh_scheme, true);

	if ((num_neighbors == 1) && ((neighbor_ids[0] > element_idx) ||
				     (neighbor_ids[0] < element_idx  && neigh_scheme[0].t8_element_level(neighbors[0]) < eclass_scheme->t8_element_level(element)
				      ))) {
	  face_neighbors.push_back(std::array<t8_locidx_t,2>{element_idx, neighbor_ids[0]});
	  double face_normal[3];
	  t8_forest_element_face_normal(forest, tree_idx, element, face_idx, face_normal);
	  face_normals.push_back(std::array<double,2>{face_normal[0], face_normal[1]});
	  face_area.push_back(t8_forest_element_face_area(forest, tree_idx, element, face_idx));
	}

	T8_FREE(neighbors);
	T8_FREE(dual_faces);
	T8_FREE(neighbor_ids);
      }

      element_idx++;
    }
  }
}
