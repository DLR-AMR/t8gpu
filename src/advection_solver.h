#ifndef ADVECTION_SOLVER_H
#define ADVECTION_SOLVER_H

#include <array>
#include <vector>
#include <string>

#include <t8.h>
#include <t8_forest/t8_forest.h>

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

private:
  void compute_edge_information();

  sc_MPI_Comm      comm;
  t8_scheme_cxx_t* scheme;
  t8_cmesh_t       cmesh;
  t8_forest_t      forest;

  // element data
  std::vector<double>               element_variable;
  std::vector<double>               element_volume;

  // face connectivity
  std::vector<std::array<t8_locidx_t,2>>    face_neighbors;

  // face data
  std::vector<std::array<double,2>> face_normals;
  std::vector<double>               face_area;

  double delta_t;

  double* device_element_variable_prev;
  double* device_element_variable_next;
  double* device_element_fluxes;
  double* device_element_volume;

  int*    device_face_neighbors;
  double* device_face_normals;
  double* device_face_area;
 };

#endif // ADVECTION_SOLVER_H
