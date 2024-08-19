/// @file compressible_euler_solver.h
/// @brief This header file declares a simple finite volume solver for
///        the compressible Euler equations

#ifndef SOLVERS_COMPRESSIBLE_EULER_SOLVER_H
#define SOLVERS_COMPRESSIBLE_EULER_SOLVER_H

#include <t8.h>
#include <thrust/device_vector.h>
#include <mesh/mesh_manager.h>

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

    /// @brief Constructor for the CompressibleEulerSolver class.
    ///
    /// This class takes ownership of the cmesh and forest given and
    /// the constructor computes the necessary face information and
    /// allocates space to store all of the variables.
    CompressibleEulerSolver(sc_MPI_Comm comm,
			    t8_scheme_cxx_t* scheme,
			    t8_cmesh_t cmesh,
			    t8_forest_t forest);

    /// @brief Destructor.
    ///
    /// This destructor frees the resouces related to the forest,
    /// cmest, variables and connectivity information.
    ~CompressibleEulerSolver();

    /// @brief This function does one step of the simulation
    ///
    /// @param delta_t The time step to advance the simulation.
    void iterate(float_type delta_t);

    /// @brief Save the density field to a vkt formatted file.
    ///
    /// @param [in] prefix   specifies the prefix used to saved the vtk
    ///             file.
    ///
    /// This member function saves the current simulation step in the
    /// vtk file format.
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
