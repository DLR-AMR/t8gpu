#include <advection_solver.h>

#include <cassert>
#include <cmath>
#include <iostream>

#include <t8.h>
#include <t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_examples.h>

#include <t8_forest/t8_forest.h>
#include <t8_schemes/t8_default/t8_default_cxx.hxx>

int adapt_callback(t8_forest_t forest, t8_forest_t forest_from, t8_locidx_t which_tree, t8_locidx_t lelement_id,
		   t8_eclass_scheme_c* ts, const int is_family, const int num_elements, t8_element_t* elements[]) {


  double center[] = {0.5, 0.5, 0.0};
  double element_centroid[3];
  t8_forest_element_centroid (forest_from, which_tree, elements[0], element_centroid);

  double distance_squared = (center[0]-element_centroid[0])*(center[0]-element_centroid[0]) + (center[1]-element_centroid[1])*(center[1]-element_centroid[1]);

  if (distance_squared < 0.2*0.2) {
    return 1;
  } else if (is_family) {

    double centroid[] = {0.0, 0.0, 0.0};
    double current_element_centroid[] = {0.0, 0.0, 0.0};
    for (size_t i=0; i<4; i++) {
      t8_forest_element_centroid (forest_from, which_tree, elements[i], current_element_centroid);
      for (size_t j=0; j<3; j++) {
	centroid[j] += current_element_centroid[j] / 4.0;
      }
    }
    double distance_squared = (center[0]-centroid[0])*(center[0]-centroid[0]) + (center[1]-centroid[1])*(center[1]-centroid[1]);
    return (distance_squared > 0.4*0.4) ? -1 : 0;
  } else {
    return 0;
  }
}

advection_solver_t::advection_solver_t() : comm(sc_MPI_COMM_WORLD),
					   cmesh(t8_cmesh_new_periodic(comm, dim)),
					   scheme(t8_scheme_new_default_cxx()),
					   forest(t8_forest_new_uniform(cmesh, scheme, level, false, comm)),
					   element_variable(t8_forest_get_local_num_elements(forest)),
					   element_volume(t8_forest_get_local_num_elements(forest)),
					   delta_t(1.0*std::pow(0.5, level+1) / sqrt(2.0)) {

  t8_forest_t new_forest = t8_forest_new_adapt(forest, adapt_callback, 0, 0, nullptr);
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

      element_variable[element_idx] = (0.5-center[0])*(0.5-center[0]) + (0.5-center[1])*(0.5-center[1]) + (0.5-center[2])*(0.5-center[2]);
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
	  face_neighbors.push_back(std::array<int,2>{element_idx, neighbor_ids[0]});
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

  cudaMalloc(&device_element_variable_prev, sizeof(double)*num_local_elements);
  cudaMalloc(&device_element_variable_next, sizeof(double)*num_local_elements);
  cudaMalloc(&device_element_fluxes, sizeof(double)*num_local_elements);
  cudaMalloc(&device_element_volume, sizeof(double)*num_local_elements);

  cudaMalloc(&device_face_neighbors, sizeof(int)*face_neighbors.size()*2);
  cudaMalloc(&device_face_normals, sizeof(double)*face_normals.size()*2);
  cudaMalloc(&device_face_area, sizeof(double)*face_normals.size()*2);

  cudaMemcpy(device_element_variable_next, element_variable.data(), element_variable.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(device_element_volume, element_volume.data(), element_volume.size()*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(device_element_fluxes, 0, element_variable.size());

  cudaMemcpy(device_face_neighbors, face_neighbors.data(), face_neighbors.size()*2*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_face_normals, face_normals.data(), face_normals.size()*2*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(device_face_area, face_area.data(), face_area.size()*2*sizeof(double), cudaMemcpyHostToDevice);
}

advection_solver_t::~advection_solver_t() {
  t8_forest_unref(&forest);
  t8_cmesh_destroy(&cmesh);

  cudaFree(device_element_variable_prev);
  cudaFree(device_element_variable_next);
  cudaFree(device_element_fluxes);
  cudaFree(device_element_volume);

  cudaFree(device_face_neighbors);
  cudaFree(device_face_normals);
  cudaFree(device_face_area);
}

__global__ static void compute_fluxes(double const* __restrict__ variable,
				      double* __restrict__ fluxes,
				      double const* __restrict__ normal,
				      double const* __restrict__ area,
				      int const* e_idx) {

  double a[2] = {0.5*sqrt(2.0), 0.5*sqrt(2.0)};

  int i = blockIdx.x * blockDim.x + threadIdx.x;

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
						double const * __restrict__ volume,
						double* __restrict__ fluxes,
						double delta_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  variable_next[i] = variable_prev[i] + delta_t/volume[i]*fluxes[i];

  fluxes[i] = 0.0;
}

void advection_solver_t::iterate() {
  std::swap(device_element_variable_next, device_element_variable_prev);

  compute_fluxes<<<face_area.size(), 1>>>(device_element_variable_prev,
					     device_element_fluxes,
					     device_face_normals,
					     device_face_area,
					     device_face_neighbors);

  explicit_euler_time_step<<<element_volume.size(),1>>>(device_element_variable_prev,
							device_element_variable_next,
							device_element_volume,
							device_element_fluxes,
							delta_t);
}

void advection_solver_t::save_vtk(const std::string& prefix) {
  cudaMemcpy(element_variable.data(), device_element_variable_next, element_variable.size()*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  t8_vtk_data_field_t vtk_data_field =  {};
  vtk_data_field.type = T8_VTK_SCALAR;
  strcpy(vtk_data_field.description, "diffusion variable");
  vtk_data_field.data = element_variable.data();

  t8_forest_write_vtk_ext(forest, prefix.c_str(), 1, 1, 1, 1, 0, 0, 0, 1, &vtk_data_field);
}
