#ifndef ADVECTION_SOLVER_H
#define ADVECTION_SOLVER_H

#include <t8.h>
#include <t8_forest/t8_forest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <shared_device_vector.h>
#include <string>

/*
 * Simple upwind finite volume advection solver
 */
class advection_solver_t {
 public:
  static constexpr size_t dim = 2;
  static constexpr t8_locidx_t min_level = 3;
  static constexpr t8_locidx_t max_level = 7;

  advection_solver_t();
  ~advection_solver_t();

  void save_vtk(const std::string& prefix);

  void iterate();
  void adapt();
  void partition();
  double compute_integral() const;
 private:
  void compute_edge_connectivity();

  sc_MPI_Comm comm;
  int rank;
  int nb_ranks;
  t8_scheme_cxx_t* scheme;
  t8_cmesh_t cmesh;
  t8_forest_t forest;

  // element data
  // TODO remove those arrays as they are only necessary in the initialization
  thrust::host_vector<double> element_variable;
  thrust::host_vector<double> element_volume;

  thrust::host_vector<double> element_refinement_criteria;

  thrust::host_vector<int> ranks;
  thrust::host_vector<t8_locidx_t> indices;
  thrust::device_vector<int> device_ranks;
  thrust::device_vector<t8_locidx_t> device_indices;

  // face connectivity
  thrust::host_vector<t8_locidx_t> face_neighbors;

  // face data
  thrust::host_vector<double> face_normals;
  thrust::host_vector<double> face_area;

  double delta_t;

  shared_device_vector<double> device_element_variable_prev;
  shared_device_vector<double> device_element_variable_next;
  shared_device_vector<double> device_element_fluxes;
  shared_device_vector<double> device_element_volume;
  thrust::device_vector<double> device_element_refinement_criteria;

  thrust::device_vector<t8_locidx_t> device_face_neighbors;
  thrust::device_vector<double> device_face_normals;
  thrust::device_vector<double> device_face_area;
};

#endif  // ADVECTION_SOLVER_H
