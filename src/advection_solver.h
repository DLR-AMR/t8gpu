/// @file advection_solver.h
/// @brief This header file declares a simple advection solver class

#ifndef ADVECTION_SOLVER_H
#define ADVECTION_SOLVER_H

#include <t8.h>
#include <t8_forest/t8_forest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <shared_device_vector.h>
#include <string>

namespace t8gpu {
  ///
  /// @brief A class that implements a simple upwind finite volume
  ///        advection solver
  ///
  class AdvectionSolver {
  public:
    static constexpr size_t dim = 2; /*! dimension of the domain */
    static constexpr t8_locidx_t min_level = 3; /*! minimum refinement level */
    static constexpr t8_locidx_t max_level = 10; /*! maximum refinement level */

    static constexpr double gamma = 1.4; /*! ratio of specific heats */
    static constexpr double cfl   = 0.7;

    /// @brief Constructor of the advection solver class
    ///
    /// @param [in]         comm specifies the MPI communicator to use
    ///
    /// This constructor initializes create a simple mesh and initializes
    /// all of its data members
    AdvectionSolver(sc_MPI_Comm comm = sc_MPI_COMM_WORLD);

    /// @brief Destructor of the advection solver class
    ///
    /// This destructor frees all the allocated device and host memory used.
    /// It is important to note that it should be called before MPI_Finalize.
    ~AdvectionSolver();

    /// @brief This function does one step of the simulation In order
    ///
    ///  To work properly on multiple MPI ranks, the method
    ///  `compute_ghost_information` must be called beforehand after
    ///  the last mesh refinement/repartitioning in order to have an
    ///  up to date ghost information. It is the responsability of the
    ///  user to do so
    void iterate(double timestep);

    /// @brief Adapt member function
    ///
    /// Adapt the mesh and enforce 2:1 balance condition. It is
    /// important to note that this operation invalidates the ghost
    /// layer. The member function `compute_ghost_information` must be
    /// called beforehand to construct an up to date ghost layer.
    void adapt();

    /// @brief Partition member function
    ///
    /// Repartition the mesh to balance the load between all MPI ranks
    /// in the comm MPI communicator. It is important to note that
    /// this operation invalidates the ghost layer. The member
    /// function `compute_ghost_information` must be called beforehand
    /// to construct an up to date ghost layer.
    void partition();

    /// @brief Ghost layer and connectivity information computation
    ///        member function
    ///
    /// This function creates the necessary ghost layer and
    /// connectivity information needed by the `iterate` member
    /// function. Upon creating of the class, it is unnecessary to
    /// call this member function as the ghost layer is already
    /// computed in the constructor for the initial mesh. However,
    /// this function may be used before the `iterate` member function
    /// if the mesh has been modified with either (or both) `adapt`,
    /// `partition` member functions after the last call to
    /// `compute_ghost_information` or the initial construction of the
    /// class.
    void compute_connectivity_information();

    /// @brief Save current quantity of interest to vkt format
    ///
    /// @param [in] prefix specifies the prefix used to saved the vtk
    ///             file
    ///
    /// This member function saves the current simulation step in the
    /// vtk file format.
    void save_vtk(const std::string& prefix) const;

    /// @brief Computes the integral of the quantity of interest
    ///
    /// @return returns the integral on all the ranks
    ///
    /// This member function computes the total integral on the domain
    /// of the quantity of interest. It can be used for sanity check
    /// in Debug mode to assert the conservativity of the scheme.
    [[nodiscard]] double compute_integral() const;

    /// @brief compute the timestep accoring to the CFL condition
    ///
    /// @return timestep
    ///
    /// This member function returns the maximum timestep according to
    /// the CFL condition and using maximum wave speed estimates. This
    /// function uses the last wave speed estimates (so the ones
    /// computed at the last step of the last timestepping)
    [[nodiscard]] double compute_timestep() const;
  private:
    sc_MPI_Comm comm;
    int rank;
    int nb_ranks;
    t8_scheme_cxx_t* scheme;
    t8_cmesh_t cmesh;
    t8_forest_t forest;

    t8_locidx_t num_local_elements;
    t8_locidx_t num_ghost_elements;
    t8_locidx_t num_local_faces;

    thrust::host_vector<int> ranks;
    thrust::host_vector<t8_locidx_t> indices;

    thrust::device_vector<int> device_ranks;
    thrust::device_vector<t8_locidx_t> device_indices;

    // host and device face connectivity data
    thrust::host_vector<t8_locidx_t> face_neighbors;
    thrust::host_vector<double> face_normals;
    thrust::host_vector<double> face_area;

    thrust::device_vector<t8_locidx_t> device_face_neighbors;
    thrust::device_vector<double> device_face_normals;
    thrust::device_vector<double> device_face_area;
    thrust::device_vector<double> device_face_speed_estimate;

    /*! enum of all variables associated to elements */
    enum VariableName {
      rho_0         = 0, /*! previous variable */
      rho_v1_0      = 1, /*! previous variable */
      rho_v2_0      = 2, /*! previous variable */
      rho_e_0       = 3, /*! previous variable */
      rho_1         = 4, /*! first SSP-RK intermediate state */
      rho_v1_1      = 5, /*! first SSP-RK intermediate state */
      rho_v2_1      = 6, /*! first SSP-RK intermediate state */
      rho_e_1       = 7, /*! first SSP-RK intermediate state */
      rho_2         = 8, /*! second SSP-RK intermediate state */
      rho_v1_2      = 9, /*! second SSP-RK intermediate state */
      rho_v2_2      = 10, /*! second SSP-RK intermediate state */
      rho_e_2       = 11, /*! second SSP-RK intermediate state */
      rho_3         = 12, /*! next variable */
      rho_v1_3      = 13, /*! next variable */
      rho_v2_3      = 14, /*! next variable */
      rho_e_3       = 15, /*! next variable */
      rho_fluxes    = 16, /*! flux contributions */
      rho_v1_fluxes = 17, /*! flux contributions */
      rho_v2_fluxes = 18, /*! flux contributions */
      rho_e_fluxes  = 19, /*! flux contributions */
      volume        = 20, /*! volume */
      nb_element_variables = 21,
    };

    VariableName rho_prev;
    VariableName rho_v1_prev;
    VariableName rho_v2_prev;
    VariableName rho_e_prev;

    VariableName rho_next;
    VariableName rho_v1_next;
    VariableName rho_v2_next;
    VariableName rho_e_next;

    /*! collection of all shared variables associated to elements */
    t8gpu::SharedDeviceVector<std::array<double, nb_element_variables>> device_element;

    thrust::host_vector<double> element_refinement_criteria;
    thrust::device_vector<double> device_element_refinement_criteria;

    void compute_edge_connectivity();
    void compute_fluxes(VariableName rho,
			VariableName rho_v1,
			VariableName rho_v2,
			VariableName rho_e);
  };
} // namespace t8gpu
#endif  // ADVECTION_SOLVER_H
