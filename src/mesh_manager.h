/// @file mesh_manager.h
/// @brief This header file declares the MeshManager class that
///        handles t8code meshes.

#ifndef MESH_MANAGER_H
#define MESH_MANAGER_H

#include <memory_manager.h>
#include <t8.h>
#include <t8_forest/t8_forest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace t8gpu {


  template<typename float_type, size_t dim>
  class MeshConnectivityAccessor {
  public:
    MeshConnectivityAccessor(const int* ranks,
			     const t8_locidx_t* indices,
			     const t8_locidx_t* face_neighbors,
			     const float_type* face_normals,
			     const float_type* face_surfaces)
      : m_ranks(ranks),
	m_indices(indices),
	m_face_neighbors(face_neighbors),
	m_face_normals(face_normals),
	m_face_surfaces(face_surfaces) {}

    [[nodiscard]] __device__ t8_locidx_t get_face_surface(int face_idx) const {
      return m_face_surfaces[face_idx];
    }

    [[nodiscard]] __device__ std::array<float_type, dim> get_face_normal(int face_idx) const {
      std::array<float_type, dim> normal {};
      for (int k=0; k<dim; k++) {
	normal[k] = m_face_normals[dim*face_idx+k];
      }
      return normal;
    }

    [[nodiscard]] __device__ std::array<t8_locidx_t, 2> get_face_neighbor_indices(int face_idx) const {
      return {m_face_neighbors[2*face_idx], m_face_neighbors[2*face_idx+1]};
    }

    [[nodiscard]] __device__ t8_locidx_t get_element_owner_rank(int element_idx) const {
      return m_ranks[element_idx];
    }

    [[nodiscard]] __device__ t8_locidx_t get_element_owner_remote_index(int element_idx) const {
      return m_indices[element_idx];
    }

  private:
    const int*         m_ranks;
    const t8_locidx_t* m_indices;
    const t8_locidx_t* m_face_neighbors;
    const float_type*  m_face_normals;
    const float_type*  m_face_surfaces;
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
  ///          └──────────────────────┘ └──────────┘      └──────────────────────┘ └──────────┘
  ///               owned elements       ghost or	        owned elements         ghost or
  ///                                 remote elements	                            remote elements
  ///
  ///          ┌───┬───┬───┬───┬───┬───┐   ╔═══╗          ┌───┬───┬───┬───┬───┬───┐
  ///  data    │   │   │   │   │   │   │   ║   ║          │   │   │   │   │   │   │
  ///          └───┴───┴───┴───┴───┴───┘   ╚═══╝          └───┴───┴───┴───┴───┴───┘
  ///                                        │                      │
  ///                                        └──────➤───────➤───────┘
  ///
  /// The class ConnectvityAccessor is prodived to access connectivity
  /// information.
  ///
  template<typename VariableType, typename StepType, size_t dim>
  class MeshManager : public MemoryManager<VariableType, StepType> {
  public:
    using float_type = typename variable_traits<VariableType>::float_type;
    using variable_index_type = typename variable_traits<VariableType>::index_type;
    static constexpr size_t nb_variables = variable_traits<VariableType>::nb_variables;

    using step_index_type = typename step_traits<StepType>::index_type;
    static constexpr size_t nb_steps = step_traits<StepType>::nb_steps;

    static constexpr t8_locidx_t max_level = 9;
    static constexpr t8_locidx_t min_level = 6;

    MeshManager(sc_MPI_Comm comm,
		t8_scheme_cxx_t* scheme,
		t8_cmesh_t cmesh,
		t8_forest_t forest);
    ~MeshManager();

    template<typename Func>
    void initialize_variables(Func func);

    void refine(const thrust::host_vector<int>& refinement_criteria, StepType step);

    void partition();

    [[nodiscard]] MeshConnectivityAccessor<float_type, dim> get_connectivity_information() const;

    [[nodiscard]] int get_num_local_elements() const;
    [[nodiscard]] int get_num_ghost_elements() const;
    [[nodiscard]] int get_num_local_faces() const;

    void compute_edge_connectivity();
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

    thrust::host_vector<float_type>   m_element_refinement_criteria;
    thrust::device_vector<float_type> m_device_refinement_criteria;

    struct UserData {
      thrust::host_vector<float_type>* element_refinement_criteria;
    };

    static int adapt_callback_iteration(t8_forest_t forest, t8_forest_t forest_from, t8_locidx_t which_tree, t8_locidx_t lelement_id, t8_eclass_scheme_c* ts,
					const int is_family, const int num_elements, t8_element_t* elements[]);
  };

  template<typename VariableType>
  __global__ void adapt_variables_and_volume(MemoryAccessorOwn<VariableType> variables_old,
					     std::array<typename variable_traits<VariableType>::float_type* __restrict__, variable_traits<VariableType>::nb_variables> variables_new,
					     typename variable_traits<VariableType>::float_type const* __restrict__ volume_old,
					     typename variable_traits<VariableType>::float_type* __restrict__       volume_new,
					     t8_locidx_t* adapt_data,
					     int nb_new_elements);


} // namespace t8gpu

#include <mesh_manager.inl>

#endif // MESH_MANAGER_H
