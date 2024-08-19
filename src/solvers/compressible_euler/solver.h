/// @file compressible_euler_solver.h
/// @brief This header file declares a simple finite volume solver for
///        the compressible Euler equations

#ifndef SOLVERS_COMPRESSIBLE_EULER_SOLVER_H
#define SOLVERS_COMPRESSIBLE_EULER_SOLVER_H

#include <t8.h>
#include <thrust/device_vector.h>
#include <mesh_manager.h>

namespace t8gpu {

  enum VariableList {
    Rho,    // density
    Rho_v1, // x-component of momentum
    Rho_v2, // y-component of momentum
    Rho_v3, // z-component of momentum
    Rho_e,  // energy
    nb_variables,
  };

  // defines the number of duplicates per variables that we need
  enum StepList {
    Step0,  // used for RK3 timestepping
    Step1,  // used for RK3 timestepping
    Step2,  // used for RK3 timestepping
    Step3,  // used for RK3 timestepping
    Fluxes, // used to store fluxes
    nb_steps
  };

  class CompressibleEulerSolver {
  public:
    static constexpr size_t dim = 3;
    using float_type = variable_traits<VariableList>::float_type;

    CompressibleEulerSolver(sc_MPI_Comm comm,
			    t8_scheme_cxx_t* scheme,
			    t8_cmesh_t cmesh,
			    t8_forest_t forest);
    ~CompressibleEulerSolver();

    void iterate(float_type delta_t);

    void save_vtk(std::string prefix) const;

  private:
    sc_MPI_Comm m_comm;

    MeshManager<VariableList, StepList, dim> m_mesh_manager;

    thrust::device_vector<float_type> m_device_face_speed_estimate;

    StepList next = Step0;
    StepList prev = Step3;
  };

} // namespace t8gpu

#endif // SOLVERS_COMPRESSIBLE_EULER_SOLVER_H
