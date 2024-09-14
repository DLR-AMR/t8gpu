/// @file compressible_euler_solver.h
/// @brief This header file declares a simple finite volume solver for
///        the compressible Euler equations

#ifndef SOLVERS_COMPRESSIBLE_EULER_SOLVER_H
#define SOLVERS_COMPRESSIBLE_EULER_SOLVER_H

#include <t8.h>
#include <t8gpu/mesh/mesh_manager.h>
#include <thrust/device_vector.h>

namespace t8gpu {

  enum VariableList {
    Rho,     // density
    Rho_v1,  // x-component of momentum
    Rho_v2,  // y-component of momentum
    Rho_v3,  // z-component of momentum
    Rho_e,   // energy
    nb_variables,
  };

  // defines the number of duplicates per variables that we need
  enum StepList {
    Step0,   // used for RK3 timestepping
    Step1,   // used for RK3 timestepping
    Step2,   // used for RK3 timestepping
    Step3,   // used for RK3 timestepping
    Fluxes,  // used to store fluxes
    nb_steps
  };

  class CompressibleEulerSolver {
   public:
    using float_type                = variable_traits<VariableList>::float_type;
    static constexpr size_t     dim = 3;
    static constexpr float_type cfl = static_cast<float_type>(0.7);

    /// @brief Constructor for the CompressibleEulerSolver class.
    ///
    /// This class takes ownership of the cmesh and forest given and
    /// the constructor computes the necessary face information and
    /// allocates space to store all of the variables.
    CompressibleEulerSolver(sc_MPI_Comm comm, t8_scheme_cxx_t* scheme, t8_cmesh_t cmesh, t8_forest_t forest);

    /// @brief Destructor.
    ///
    /// This destructor frees the resouces related to the forest,
    /// cmest, variables and connectivity information.
    ~CompressibleEulerSolver();

    /// @brief This function does one step of the simulation
    ///
    /// @param delta_t The time step to advance the simulation.
    void iterate(float_type delta_t);

    /// @brief This function refines or coarsens mesh elements
    ///        according to an approximation of the gradient.
    void adapt();

    /// @brief Save the density field to a vkt formatted file.
    ///
    /// @param [in] prefix   specifies the prefix used to saved the vtk
    ///             file.
    ///
    /// This member function saves the current simulation step in the
    /// vtk file format.
    void save_conserved_variables_to_vtk(std::string prefix) const;

    /// @brief Computes the integral of the quantity of interest.
    ///
    /// @return returns the integral on all the ranks.
    ///
    /// This member function computes the total integral on the domain
    /// of the quantity of interest. It can be used for sanity check
    /// in Debug mode to assert the conservativity of the scheme.
    [[nodiscard]] float_type compute_integral() const;

    /// @brief compute the timestep accoring to the CFL condition.
    ///
    /// @return timestep.
    ///
    /// This member function returns the maximum timestep according to
    /// the CFL condition and using maximum wave speed estimates. This
    /// function uses the last wave speed estimates (so the ones
    /// computed at the last step of the last timestepping).
    [[nodiscard]] float_type compute_timestep() const;

   private:
    sc_MPI_Comm m_comm;

    MeshManager<VariableList, StepList, dim> m_mesh_manager;

    /** We compute wave speed estimates in the Roe-averaged state */
    thrust::device_vector<float_type> m_device_face_speed_estimate;

    /** The most up to date timestep is stored in the 'next´ step, to
        advance the simulation, the 'next´ and 'prev´ step pointers
        are swaped */
    StepList next = Step0;
    StepList prev = Step3;
  };

}  // namespace t8gpu

#endif  // SOLVERS_COMPRESSIBLE_EULER_SOLVER_H
