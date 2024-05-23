#include <array>
#include <iostream>
#include <vector>

#include <t8.h>
#include <t8_cmesh.h>
#include <t8_cmesh_vtk_writer.h>
#include <t8_cmesh/t8_cmesh_examples.h>

#include <t8_forest/t8_forest.h>
#include <t8_schemes/t8_default/t8_default_cxx.hxx>

#define NUM_FACES 4

int main(int argc, char* argv[]) {

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  for (int device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d:%s has compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
  }

  int mpiret = sc_MPI_Init(&argc, &argv);
  SC_CHECK_MPI(mpiret);

  sc_init(sc_MPI_COMM_WORLD, 1, 1, NULL, SC_LP_ESSENTIAL);
  t8_init(SC_LP_PRODUCTION);

  sc_MPI_Comm comm = sc_MPI_COMM_WORLD;
  // t8_cmesh_t coarse_mesh = t8_cmesh_new_hypercube(T8_ECLASS_HEX, comm, 0, 0, 0);
  t8_cmesh_t coarse_mesh = t8_cmesh_new_periodic(comm, 2);

  t8_scheme_cxx_t* scheme = t8_scheme_new_default_cxx();
  const int level = 3;
  t8_forest_t forest = t8_forest_new_uniform(coarse_mesh, scheme, level, false, comm);

  t8_locidx_t num_local_elements = t8_forest_get_local_num_elements(forest);

  std::vector<double> element_data(num_local_elements);
  std::vector<std::array<t8_locidx_t, NUM_FACES>> element_neighbor_indices(num_local_elements);

  t8_locidx_t num_local_trees = t8_forest_get_num_local_trees(forest);
  t8_locidx_t element_idx = 0;
  for (t8_locidx_t tree_idx = 0; tree_idx < num_local_trees; tree_idx++) {
    t8_locidx_t num_elements_in_tree = t8_forest_get_tree_num_elements(forest, tree_idx);
    for (t8_locidx_t tree_element_idx = 0; tree_element_idx < num_elements_in_tree; tree_element_idx++) {
      const t8_element_t* element = t8_forest_get_element_in_tree(forest, tree_idx, tree_element_idx);

      double center[3];
      t8_forest_element_centroid(forest, tree_idx, element, center);

      element_data[element_idx] = (0.5-center[0])*(0.5-center[0]) + (0.5-center[1])*(0.5-center[1]) + (0.5-center[2])*(0.5-center[2]);

      for (size_t face_idx=0; face_idx < NUM_FACES; face_idx++) {
	int num_neighbors;
	int *dual_faces;
	t8_locidx_t *neighbor_ids;
	t8_element_t **neighbors;
	t8_eclass_scheme_c *neigh_scheme;

	t8_forest_leaf_face_neighbors(forest, tree_idx, element, &neighbors, face_idx, &dual_faces, &num_neighbors,
				      &neighbor_ids, &neigh_scheme, 1);
	element_neighbor_indices[element_idx][face_idx] = neighbor_ids[0];

	T8_FREE(neighbors);
	T8_FREE(dual_faces);
	T8_FREE(neighbor_ids);
      }

      element_idx++;
    }
  }

  t8_vtk_data_field_t vtk_data_field =  {};
  vtk_data_field.type = T8_VTK_SCALAR;
  strcpy(vtk_data_field.description, "diffusion variable");
  vtk_data_field.data = element_data.data();

  t8_forest_write_vtk_ext(forest, "initial_data", 1, 1, 1, 1, 0, 0, 0, 1, &vtk_data_field);

  t8_forest_unref(&forest);
  t8_cmesh_destroy(&coarse_mesh);

  sc_finalize();

  mpiret = sc_MPI_Finalize();
  SC_CHECK_MPI(mpiret);

  return 0;
}
