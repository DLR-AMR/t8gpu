#include <advection_solver.h>

#include <cassert>
#include <cmath>
#include <iostream>

#include <t8.h>
#include <t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_examples.h>

#include <t8_forest/t8_forest.h>
#include <t8_schemes/t8_default/t8_default_cxx.hxx>

static constexpr size_t num_faces = 4;

static constexpr double normals[4][2] = {
					 {-1.0,  0.0},
					 { 1.0,  0.0},
					 { 0.0, -1.0},
					 { 0.0,  1.0}
};

advection_solver_t::advection_solver_t() : comm(sc_MPI_COMM_WORLD),
					   cmesh(t8_cmesh_new_periodic(comm, dim)),
					   scheme(t8_scheme_new_default_cxx()),
					   forest(t8_forest_new_uniform(cmesh, scheme, level, false, comm)),
					   element_data(t8_forest_get_local_num_elements(forest)),
					   delta_x(std::pow(0.5, level)),
					   delta_t(1.0*delta_x) {

  t8_locidx_t num_local_elements = element_data.size();

  t8_locidx_t num_local_trees = t8_forest_get_num_local_trees(forest);
  t8_locidx_t element_idx = 0;
  for (t8_locidx_t tree_idx = 0; tree_idx < num_local_trees; tree_idx++) {
    t8_locidx_t num_elements_in_tree = t8_forest_get_tree_num_elements(forest, tree_idx);
    for (t8_locidx_t tree_element_idx = 0; tree_element_idx < num_elements_in_tree; tree_element_idx++) {
      const t8_element_t* element = t8_forest_get_element_in_tree(forest, tree_idx, tree_element_idx);

      double center[3];
      t8_forest_element_centroid(forest, tree_idx, element, center);

      element_data[element_idx] = (0.5-center[0])*(0.5-center[0]) + (0.5-center[1])*(0.5-center[1]) + (0.5-center[2])*(0.5-center[2]);

      for (size_t face_idx=0; face_idx < num_faces; face_idx++) {
	int num_neighbors;
	int* dual_faces;
	t8_locidx_t* neighbor_ids;
	t8_element_t** neighbors;
	t8_eclass_scheme_c* neigh_scheme;

	t8_forest_leaf_face_neighbors(forest, tree_idx, element, &neighbors, face_idx, &dual_faces, &num_neighbors,
				      &neighbor_ids, &neigh_scheme, 1);

	if (neighbor_ids[0] > element_idx) {
	  face_neighbors.push_back(std::array<int,2>{element_idx, neighbor_ids[0]});
	  face_normals.push_back(std::array<double,2>{normals[face_idx][0], normals[face_idx][1]});
	}

	T8_FREE(neighbors);
	T8_FREE(dual_faces);
	T8_FREE(neighbor_ids);
      }

      element_idx++;
    }
  }

  cudaMalloc(&device_element_data_input, sizeof(double)*num_local_elements);
  cudaMalloc(&device_element_data_output, sizeof(double)*num_local_elements);
  cudaMalloc(&device_face_neighbors, sizeof(int)*face_neighbors.size()*2);
  cudaMalloc(&device_face_normals, sizeof(double)*face_normals.size()*2);

  cudaMemcpy(device_element_data_output, element_data.data(), num_local_elements*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(device_face_neighbors, face_neighbors.data(), face_neighbors.size()*2*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_face_normals, face_normals.data(), face_normals.size()*2*sizeof(double), cudaMemcpyHostToDevice);
}

advection_solver_t::~advection_solver_t() {
  t8_forest_unref(&forest);
  t8_cmesh_destroy(&cmesh);

  cudaFree(device_element_data_input);
  cudaFree(device_element_data_output);
  cudaFree(device_face_neighbors);
  cudaFree(device_face_normals);
}

__global__ static void exchange_fluxes(double const* __restrict__ in,
				       double* __restrict__ out,
				       double const* __restrict__ normal,
				       int const* e_idx,
				       double delta_t, double delta_x) {

  double a[2] = {0.5*sqrt(2.0), 0.5*sqrt(2.0)};

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  double flux = delta_t/delta_x*(a[0]*normal[2*i]+a[1]*normal[2*i+1]);

  if (flux > 0.0) {
    flux *= in[e_idx[2*i]];
  } else {
    flux *= in[e_idx[2*i+1]];
  }

  atomicAdd(&out[e_idx[2*i]], -flux);
  atomicAdd(&out[e_idx[2*i+1]], flux);
}

void advection_solver_t::iterate() {
  cudaMemcpy(device_element_data_input, device_element_data_output, element_data.size()*sizeof(double), cudaMemcpyDeviceToDevice);
  exchange_fluxes<<<face_neighbors.size(), 1>>>(device_element_data_input,
						device_element_data_output,
						device_face_normals,
						device_face_neighbors,
						delta_t, delta_x);
}

void advection_solver_t::save_vtk(const std::string& prefix) {
  cudaMemcpy(element_data.data(), device_element_data_output, element_data.size()*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  t8_vtk_data_field_t vtk_data_field =  {};
  vtk_data_field.type = T8_VTK_SCALAR;
  strcpy(vtk_data_field.description, "diffusion variable");
  vtk_data_field.data = element_data.data();

  t8_forest_write_vtk_ext(forest, prefix.c_str(), 1, 1, 1, 1, 0, 0, 0, 1, &vtk_data_field);
}
