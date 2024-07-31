/// @file compressible_euler_solver.h
/// @brief This header file declares a simple finite volume solver for
///        the compressible Euler equations

#ifndef COMPRESSIBLE_EULER_SOLVER_H
#define COMPRESSIBLE_EULER_SOLVER_H

#include <cuda/std/array>
#include <t8.h>
#include <t8_forest/t8_forest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <shared_device_vector.h>
#include <string>

namespace t8gpu {
  ///
  /// @brief A class that implements a simple finite volume solver for
  /// the compressible Euler equations
  ///
  class CompressibleEulerSolver {
  public:
    using float_type = float;
    static constexpr size_t dim = 3; /*! dimension of the domain */
    static constexpr t8_locidx_t min_level = 6;  /*! minimum refinement level */
    static constexpr t8_locidx_t max_level = 12; /*! maximum refinement level */

    static constexpr float_type gamma = float_type{1.4}; /*! ratio of specific heats */
    static constexpr float_type cfl   = float_type{0.7};

    /// @brief Constructor of the advection solver class
    ///
    /// @param [in]         comm specifies the MPI communicator to use
    ///
    /// This constructor initializes create a simple mesh and initializes
    /// all of its data members
    CompressibleEulerSolver(sc_MPI_Comm comm = sc_MPI_COMM_WORLD);

    /// @brief Destructor of the advection solver class
    ///
    /// This destructor frees all the allocated device and host memory used.
    /// It is important to note that it should be called before MPI_Finalize.
    ~CompressibleEulerSolver();

    /// @brief This function does one step of the simulation In order
    ///
    ///  To work properly on multiple MPI ranks, the method
    ///  `compute_ghost_information` must be called beforehand after
    ///  the last mesh refinement/repartitioning in order to have an
    ///  up to date ghost information. It is the responsability of the
    ///  user to do so
    void iterate(float_type timestep);

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
    [[nodiscard]] float_type compute_integral() const;

    /// @brief compute the timestep accoring to the CFL condition
    ///
    /// @return timestep
    ///
    /// This member function returns the maximum timestep according to
    /// the CFL condition and using maximum wave speed estimates. This
    /// function uses the last wave speed estimates (so the ones
    /// computed at the last step of the last timestepping)
    [[nodiscard]] float_type compute_timestep() const;
  private:
    sc_MPI_Comm      m_comm;
    int              m_rank;
    int              m_nb_ranks;
    t8_scheme_cxx_t* m_scheme;
    t8_cmesh_t       m_cmesh;
    t8_forest_t      m_forest;

    t8_locidx_t m_num_local_elements;
    t8_locidx_t m_num_ghost_elements;
    t8_locidx_t m_num_local_faces;

    thrust::host_vector<int>           m_ranks;
    thrust::device_vector<int>         m_device_ranks;
    thrust::host_vector<t8_locidx_t>   m_indices;
    thrust::device_vector<t8_locidx_t> m_device_indices;

    // host and device face connectivity data
    thrust::host_vector<t8_locidx_t>   m_face_neighbors;
    thrust::device_vector<t8_locidx_t> m_device_face_neighbors;
    thrust::host_vector<float_type>    m_face_normals;
    thrust::device_vector<float_type>  m_device_face_normals;
    thrust::host_vector<float_type>    m_face_area;
    thrust::device_vector<float_type>  m_device_face_area;
    thrust::device_vector<float_type>  m_device_face_speed_estimate;

    // defines the set of conserved variables
    enum VariableName {
      rho,    // density
      rho_v1, // x-component of momentum
      rho_v2, // y-component of momentum
      rho_v3, // z-component of momentum
      rho_e,  // energy
      nb_conserved_variables
    };

    // defines the number of duplicates per variables that we need
    enum StepName {
      step0,  // used for RK3 timestepping
      step1,  // used for RK3 timestepping
      step2,  // used for RK3 timestepping
      step3,  // used for RK3 timestepping
      fluxes, // used to store fluxes
      nb_steps
    };

    StepName prev = step0;
    StepName next = step3;

    [[nodiscard]] static int get_var(StepName step_name, VariableName var_name) {
      return var_name + step_name*nb_conserved_variables;
    }

    [[nodiscard]] cuda::std::array<float_type* __restrict__, nb_conserved_variables> get_own_vars(StepName step_name) {
      cuda::std::array<float_type* __restrict__, nb_conserved_variables> vars {};
      // TODO: this could be done with a object that holds only one pointer but has __device__ operator[] to do the correct strided access into the shared device vector m_device_array member

      for (int k=0; k<nb_conserved_variables; k++) {
	vars[k] = m_device_element.get_own(get_var(step_name, static_cast<VariableName>(k)));
      }
      return vars;
    }

    [[nodiscard]] static int get_vol() {
      return nb_steps*nb_conserved_variables;
    }

    constexpr static size_t nb_element_variables = nb_conserved_variables * nb_steps + 1;

    /*! collection of all shared variables associated to elements */
    t8gpu::SharedDeviceVector<std::array<float_type, nb_element_variables>> m_device_element;

    thrust::host_vector<float_type>   m_element_refinement_criteria;
    thrust::device_vector<float_type> m_device_element_refinement_criteria;

    void compute_edge_connectivity();
    void compute_fluxes(StepName step_name);

    template<typename ft = float_type>
    void save_vtk_impl(const std::string& prefix) const;
  };
} // namespace t8gpu
#endif  // COMPRESSIBLE_EULER_SOLVER_H
