/// @file subgrid_mesh_manager.h
/// @brief This header file declares the SubgridMeshManager class that
///        handles t8code meshes as well as the
///        MeshConnectivityAccessor helper class.

#ifndef MESH_SUBGRID_MESH_MANAGER_H
#define MESH_SUBGRID_MESH_MANAGER_H

#include <t8.h>
#include <t8_forest/t8_forest.h>
#include <t8_vtk.h>
#include <t8gpu/memory/subgrid_memory_manager.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <memory>
#include <vector>

namespace t8gpu {

  /// Forward declaration of MeshManager class needed to make it a
  /// friend class of the MeshConnectivityAccessor{Own,All}.
  template<typename VariableType, typename StepType, typename SubgridType>
  class SubgridMeshManager;

  ///
  /// @brief This class gives access to face connectivity information
  ///        on the GPU.
  ///
  template<typename float_type, typename SubgridType>
  class SubgridMeshConnectivityAccessor {
    /// Friend class that can instantiate this a class.
    template<typename VT, typename ST, typename SubgridType_>
    friend class SubgridMeshManager;

    constexpr static int dim = SubgridType::rank;

   public:
    /// @brief copy constructor.
    SubgridMeshConnectivityAccessor(SubgridMeshConnectivityAccessor const& other) = default;

    /// @brief assignment constructor.
    SubgridMeshConnectivityAccessor& operator=(SubgridMeshConnectivityAccessor const& other) = default;

    /// @brief get the number of local faces.
    ///
    /// @return the number of faces.
    [[nodiscard]] __device__ __host__ inline t8_locidx_t get_num_local_faces() const { return m_num_local_faces; }

    /// @brief get the number of local boundary faces.
    ///
    /// @return the number of boundary faces.
    [[nodiscard]] __device__ __host__ inline t8_locidx_t get_num_local_boundary_faces() const {
      return m_num_local_boundary_faces;
    }

    /// @brief get the surface of a face.
    ///
    /// @param face_idx the face index.
    ///
    /// @return the surface of the face.
    [[nodiscard]] __device__ inline float_type get_face_surface(int face_idx) const {
      return m_face_surfaces[face_idx];
    }

    /// @brief get the surface of a boundary face.
    ///
    /// @param face_idx the boundary face index.
    ///
    /// @return the surface of the boundary face.
    [[nodiscard]] __device__ inline float_type get_boundary_face_surface(int face_idx) const {
      return m_face_surfaces[m_num_local_faces + face_idx];
    }

    /// @brief get the normal of a face.
    ///
    /// @param face_idx the face index.
    ///
    /// @return the normal of the face specified as a dim-element
    ///         array.
    [[nodiscard]] __device__ inline std::array<float_type, dim> get_face_normal(int face_idx) const {
      std::array<float_type, dim> normal{};
      for (int k = 0; k < dim; k++) {
        normal[k] = m_face_normals[dim * face_idx + k];
      }
      return normal;
    }

    /// @brief get the normal of a boundary face.
    ///
    /// @param face_idx the boundary face index.
    ///
    /// @return the normal of the boundary face specified as a
    ///         dim-element array.
    [[nodiscard]] __device__ inline std::array<float_type, dim> get_boundary_face_normal(int face_idx) const {
      std::array<float_type, dim> normal{};
      for (int k = 0; k < dim; k++) {
        normal[k] = m_face_normals[dim * (m_num_local_faces + face_idx) + k];
      }
      return normal;
    }

    /// @brief get index of face neighbor elements.
    ///
    /// @param face_idx the face index.
    ///
    /// @return the indices of the 2 face neighbors. Those indices can
    ///         refer to either owned elements (if the index is < than
    ///         the local number of elements), or ghost elements (if
    ///         the face is at a boundary between two domains owned by
    ///         different ranks). To fetch data for those elements,
    ///         you need to retrieve the remove indices and ranks
    ///         using the following member functions.
    ///
    /// @warning It is important to note that an index returned by
    ///          this function is a local index and not a remote
    ///          index. If it is a ghost element, to get the remote
    ///          index into the rank that owns the ghost element, use
    ///          the get_element_owner_remote_rank member function to
    ///          get the remote index and get_element_owner_rank to
    ///          get the owning rank.
    [[nodiscard]] __device__ inline std::array<t8_locidx_t, 2> get_face_neighbor_indices(int face_idx) const {
      return {m_face_neighbors[2 * face_idx], m_face_neighbors[2 * face_idx + 1]};
    }

    /// @brief get index of the local element neighbors of a face.
    ///
    /// @param face_idx the face index.
    ///
    /// @return the index of the face neighbor element. This index is
    ///         a local index as boundary faces are owned by the rank
    ///         that owns its only neighbor element. That's why the
    ///         member function get_boundary_face_neighbors_rank does
    ///         not exist.
    [[nodiscard]] __device__ inline t8_locidx_t get_boundary_face_neighbor_index(int face_idx) const {
      return m_face_neighbors[2 * m_num_local_faces + face_idx];
    }

    /// @brief get the owning rank of an element.
    ///
    /// @param element_idx local element index.
    ///
    /// @return The rank that owns the element.
    [[nodiscard]] __device__ inline t8_locidx_t get_element_owner_rank(int element_idx) const {
      return m_ranks[element_idx];
    }

    /// @brief get the remote index of an element.
    ///
    /// @param element_idx local element index.
    ///
    /// @return The remote index into the data array of the owning
    ///         rank. If element_idx refers to a element owned by the
    ///         current rank, the function returns the same
    ///         index. Otherwise, if the element_idx refers to a ghost
    ///         element, it returns the offset into the owning data
    ///         array of the ghost element.
    [[nodiscard]] __device__ inline t8_locidx_t get_element_owner_remote_index(int element_idx) const {
      return m_indices[element_idx];
    }

   private:
    int const*         m_ranks;
    t8_locidx_t const* m_indices;
    t8_locidx_t const* m_face_neighbors;
    float_type const*  m_face_normals;
    float_type const*  m_face_surfaces;
    t8_locidx_t const  m_num_local_faces;
    t8_locidx_t const  m_num_local_boundary_faces;

    SubgridMeshConnectivityAccessor(int const*         ranks,
                                    t8_locidx_t const* indices,
                                    t8_locidx_t const* face_neighbors,
                                    float_type const*  face_normals,
                                    float_type const*  face_surfaces,
                                    t8_locidx_t const  num_local_faces,
                                    t8_locidx_t const  num_local_boundary_faces)
        : m_ranks{ranks},
          m_indices{indices},
          m_face_neighbors{face_neighbors},
          m_face_normals{face_normals},
          m_face_surfaces{face_surfaces},
          m_num_local_faces{num_local_faces},
          m_num_local_boundary_faces{num_local_boundary_faces} {}
  };

  ///
  /// @brief class that represents a distributed mesh and provides mesh
  ///        operations such as refinement, partition routines.
  ///
  ///
  /// The MeshManager template class provides a convenient way to
  /// access face neighbor element data. Each rank stores data
  /// associated to the element that it owns. However, to be able to
  /// iterate over the faces, we need a way to retrieve data for
  /// elements that may not be owned by our own rank (for instance for
  /// an egde at the boudary between two subdomains owned by different
  /// ranks). In order to to so, the MeshManager class provides a way
  /// to retrieve ghost information using the rank and remote index of
  /// the data we want to access. Despite requiring more GPU memory
  /// space (to store the owned element ranks and indices which are
  /// known in advance), this data management strategy allows us to
  /// have a simple uniform access pattern for owned element and ghost
  /// element which is prefered for GPU codes. Moreover, this class
  /// stores connectivity information in the form of rank and remote
  /// index for each face neighbor elements as well as normal as face
  /// surface data. Here is a diagram of the data layout:
  ///
  ///
  ///                                      ghost layer bookkeeping
  ///
  ///                          rank 0                                      rank 1
  ///          ┌───┬───┬───┬───┬───┬───┰───┬───┬───┐	   ┌───┬───┬───┬───┬───┬───┰───┬───┬───┐
  ///  ranks   │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 ┃ 2 │ 1 │ 3 │	   │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 ┃ 0 │ 4 │ 3 │
  ///          └───┴───┴───┴───┴───┴───┸───┴───┴───┘	   └───┴───┴───┴───┴───┴───┸───┴───┴───┘
  ///
  /// (remote) ┌───┬───┬───┬───┬───┬───┰───┬───┬───┐	   ┌───┬───┬───┬───┬───┬───┰───┬───┬───┐
  ///  indices │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 ┃ 0 │ 2 │ 1 │	   │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 ┃ 2 │ 1 │ 0 │
  ///          └───┴───┴───┴───┴───┴───┸───┴───┴───┘	   └───┴───┴───┴───┴───┴───┸───┴───┴───┘
  ///
  ///          └──────────────────────┘ └──────────┘       └──────────────────────┘ └──────────┘
  ///               owned elements       ghost or	         owned elements         ghost or
  ///                                 remote elements	                             remote elements
  ///
  ///          ┌───┬───┬───┬───┬───┬───┐   ╔═══╗           ┌───┬───┬───┬───┬───┬───┐
  ///  data    │   │   │   │   │   │   │   ║   ║           │   │   │   │   │   │   │
  ///          └───┴───┴───┴───┴───┴───┘   ╚═══╝           └───┴───┴───┴───┴───┴───┘
  ///                                        │                       │
  ///                                        └───────➤───────➤───────┘
  ///
  /// The class MeshConnectvityAccessor is prodived to access
  /// connectivity information.
  ///
  template<typename VariableType, typename StepType, typename SubgridType>
  class SubgridMeshManager : public SubgridMemoryManager<VariableType, StepType, SubgridType> {
   public:
    using float_type                  = typename variable_traits<VariableType>::float_type;
    using variable_index_type         = typename variable_traits<VariableType>::index_type;
    static constexpr int nb_variables = variable_traits<VariableType>::nb_variables;
    static constexpr int dim          = SubgridType::rank;

    using step_index_type            = typename step_traits<StepType>::index_type;
    static constexpr size_t nb_steps = step_traits<StepType>::nb_steps;

    static constexpr t8_locidx_t min_level = 1;
    static constexpr t8_locidx_t max_level = 5;

    /// @brief Constructor of the MeshManager class.
    ///
    /// @param [in]   comm   specifies the MPI communicator to use.
    /// @param [in]   scheme specifies the t8code scheme used.
    /// @param [in]   cmesh  specifies the t8code cmesh used.
    /// @param [in]   forest specifies the t8code forest.
    ///
    /// This constructor initializes a MeshManager class taking
    /// ownership of the coarse mesh cmesh and forest.
    SubgridMeshManager(sc_MPI_Comm comm, t8_scheme_cxx_t* scheme, t8_cmesh_t cmesh, t8_forest_t forest);

    /// @brief Destructor of the MeshManager class.
    ///
    /// This destructor releases all of the resources owned,
    /// i.e. cmesh, forest and user data associated to the forest.
    ~SubgridMeshManager();

    /// @brief member function to initialize the variables.
    ///
    /// @param [in] func specifies the function used to initialize the
    ///                  variables.
    ///
    /// The Func function can either be a lambda, function pointer
    /// (something that is callable). It must have the following
    /// signature:
    ///
    /// void func(MemoryAccessorOwn<VariableList>& accessor,
    ///           t8_forest_t forest,
    ///           t8_locidx_t tree_idx,
    ///           const t8_element_t* element,
    ///           t8_locidx_t e_idx);
    template<typename Func>
    void initialize_variables(Func func);

    /// @brief adapt the mesh according to a user  specified criteria.
    ///
    /// @param [in] refinement_criteria specifies the refinement, 0 to
    ///                                 mean do not refine or coarsen,
    ///                                 1 to refine and -1 to coarsen.
    ///
    /// @param [in] step the step for which the variables are copied
    ///                  over.
    ///
    /// It is important to note that after such an operation, it is
    /// necessary to call compute_connectivity_information to
    /// recompute the connectivity information and to discard any
    /// other object related to connectivity initialized before the
    /// refinement.
    void adapt(thrust::host_vector<float_type> const& refinement_criteria, step_index_type step);

    /// @brief repartition the elements among the ranks.
    ///
    /// @param [in] step the step for which the variables are copied.
    ///
    /// It is important to note that after such an operation, it is
    /// necessary to call compute_connectivity_information to
    /// recompute the connectivity information and to discard any
    /// other object related to connectivity initialized before the
    /// partition.
    void partition(step_index_type step);

    /// @brief Ghost layer and connectivity information computation
    ///        member function.
    ///
    /// This function creates the necessary ghost layer and
    /// connectivity information needed. Upon creation of the class,
    /// it is unnecessary to call this member function as the ghost
    /// layer is already computed in the constructor for the initial
    /// mesh. However, this function may be used before any operation
    /// requiring connectivity information if the mesh has been
    /// modified with either (or both) `refine`, `partition` member
    /// functions after the last call to `compute_ghost_information`
    /// or the initial construction of the class.
    void compute_connectivity_information();

    /// @brief Save current quantity of interest to vkt format.
    ///
    /// @param [in] step     specifies the step from which to choose the
    ///                      variable.
    /// @param [in] variable specifies the variable.
    /// @param [in] prefix   specifies the prefix used to saved the vtk
    ///             file.
    ///
    /// This member function saves the current simulation step in the
    /// vtk file format. To save multiple variables in the same VTK
    /// file, use the member function save_variables_to_vtk instead.
    void save_variable_to_vtk(step_index_type step, variable_index_type variable, std::string const& prefix) const;

    /// @brief Host copy variable memory owning wrapper around the
    /// type t8_vtk_data_field_t.
    ///
    /// This class represents a vector or scalar field stored on the
    /// CPU of one of the variables. This class can only be created by
    /// the parent class MeshManager through the member function
    /// get_host_{scalar,vector}_variable and used to save them in a
    /// VTK file using the member function save_variables_to_vtk.
    /// From the perspective of the user, this class is opaque
    /// (meaning the user cannot construct nor modify any instance of
    /// this class). One of the objective of this class is to stick
    /// with the RAII paradigm while providing a way for the user to
    /// manually select the variable that should be exported into a
    /// VTK file.
    class HostVariableInfo {
      friend t8gpu::SubgridMeshManager<VariableType, StepType, SubgridType>;

      /// @brief Constructor only accessible through the MeshManager
      ///        class. This constructor takes ownership of the data
      ///        array.
      HostVariableInfo(t8_vtk_data_type_t data_type, std::unique_ptr<double[]>&& data, std::string const& name)
          : m_data{std::move(data)} {
        m_vtk_data_field_info_struct.type = data_type;
        m_vtk_data_field_info_struct.data = m_data.get();
        std::strncpy(m_vtk_data_field_info_struct.description, name.c_str(), BUFSIZ - 1);
      }

     public:
      /// @brief default constructor needed to be able to construct
      ///        vectors of HostVariableInfo structs.
      HostVariableInfo() = default;

      /// @brief default move constructor needed to be able to
      ///        construct vectors of HostVariableInfo structs.
      HostVariableInfo(HostVariableInfo&&) = default;

      /// @brief default destructor.
      ~HostVariableInfo() = default;

     private:
      /** unique pointer owning variable data */
      std::unique_ptr<double[]> m_data;
      /** vtk struct referencing m_data */
      t8_vtk_data_field_t m_vtk_data_field_info_struct;
    };

    /// @brief Copy a scalar variable on the CPU in order to later
    ///        save it as a VTK file.
    ///
    /// @param [in] step     specifies the step from which to choose the
    ///                      variable.
    /// @param [in] variable specifies the variable to get.
    /// @param [in] name     a name to reference the variable.
    [[nodiscard]] HostVariableInfo get_host_scalar_variable(step_index_type     step,
                                                            variable_index_type variable,
                                                            std::string const&  name) const;

    /// @brief Copy a vectorial variable on the CPU in order to later
    ///        save it as a VTK file.
    ///
    /// @param [in] step     specifies the step from which to choose the
    ///                      variable.
    /// @param [in] variable specifies the variables to aggregate into
    ///                      a vector field.
    /// @param [in] name     a name to reference the variable.
    [[nodiscard]] HostVariableInfo get_host_vector_variable(step_index_type                    step,
                                                            std::array<variable_index_type, 3> variable,
                                                            std::string const&                 name) const;

    /// @brief Save a list of variables to a VTK file.
    ///
    /// @param [in] host_variables specifies the list of variables to save.
    /// @param [in] prefix         specifies the prefix used to saved the vtk.
    void save_variables_to_vtk(std::vector<HostVariableInfo> host_variables, std::string const& prefix) const;

    /// @brief get a connectivity struct that can be passed and used
    ///        on the GPU.
    [[nodiscard]] SubgridMeshConnectivityAccessor<float_type, SubgridType> get_connectivity_information() const;

    /// @brief get the number of elements owned by this rank.
    [[nodiscard]] t8_locidx_t get_num_local_elements() const;

    /// @brief get the number of ghost elements for this rank.
    [[nodiscard]] t8_locidx_t get_num_ghost_elements() const;

    /// @brief get the number of faces owned by this rank.
    [[nodiscard]] t8_locidx_t get_num_local_faces() const;

    /// @brief get the number of boundary faces owned by this rank.
    [[nodiscard]] t8_locidx_t get_num_local_boundary_faces() const;

   private:
    /// We make the base class resize member function private.
    using SubgridMemoryManager<VariableType, StepType, SubgridType>::resize;

    sc_MPI_Comm      m_comm;
    int              m_rank;
    int              m_nb_ranks;
    t8_scheme_cxx_t* m_scheme;
    t8_cmesh_t       m_cmesh;
    t8_forest_t      m_forest;

    t8_locidx_t m_num_local_elements;
    t8_locidx_t m_num_ghost_elements;
    t8_locidx_t m_num_local_faces;
    t8_locidx_t m_num_local_boundary_faces;

    // TODO: remove CPU copied variables.
    thrust::host_vector<int>           m_ranks;
    thrust::device_vector<int>         m_device_ranks;
    thrust::host_vector<t8_locidx_t>   m_indices;
    thrust::device_vector<t8_locidx_t> m_device_indices;

   public:                                                             // for debugging purposes. Remove this ASAP.
    thrust::device_vector<t8_locidx_t> m_device_face_level_difference; /** inner faces neighbor elements */
    thrust::device_vector<t8_locidx_t>
        m_device_face_neighbor_offset;                          /** face anchor position in neighboring elements */
   private:                                                     // for debugging purposes. Remove this ASAP.
    thrust::device_vector<t8_locidx_t> m_device_face_neighbors; /** inner faces neighbor elements */
    thrust::device_vector<float_type>  m_device_face_normals;   /** inner and boundary faces normals */
    thrust::device_vector<float_type>  m_device_face_area;      /** inner and boundary faces area */

    thrust::host_vector<float_type> m_element_refinement_criteria;

    /// @brief Struct to store user data passed to t8code callback
    ///        functions.
    struct UserData {
      thrust::host_vector<float_type>* element_refinement_criteria;
    };

    static int adapt_callback_iteration(t8_forest_t         forest,
                                        t8_forest_t         forest_from,
                                        t8_locidx_t         which_tree,
                                        t8_locidx_t         lelement_id,
                                        t8_eclass_scheme_c* ts,
                                        int const           is_family,
                                        int const           num_elements,
                                        t8_element_t*       elements[]);
  };
}  // namespace t8gpu

#include "subgrid_mesh_manager.inl"

#endif  // MESH_SUBGRID_MESH_MANAGER_H
