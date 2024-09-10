#ifndef EXAMPLES_RAYLEIGH_TAYLOR_KERNELS_H
#define EXAMPLES_RAYLEIGH_TAYLOR_KERNELS_H

#include <t8gpu/memory/subgrid_memory_manager.h>
#include <t8gpu/mesh/subgrid_mesh_manager.h>

#include "solver.h"

namespace t8gpu {

  /// @brief This kernel computes the kepes flux at every inner faces
  ///        within subgrid elements and adds every contribution to
  ///        the fluxes variables. The flux variable must be
  ///        set beforehand to zeros.
  ///
  /// @param [in]  variables variables used to compute the fluxes.
  /// @param [out] fluxes    per subgrid element fluxes variables.
  /// @param [in]  volume    volumes of each elements.
  __global__ void compute_inner_fluxes(t8gpu::SubgridMemoryAccessorOwn<t8gpu::VariableList, t8gpu::SubgridCompressibleEulerSolver::subgrid_type> variables,
				       t8gpu::SubgridMemoryAccessorOwn<t8gpu::VariableList, t8gpu::SubgridCompressibleEulerSolver::subgrid_type> fluxes,
				       t8gpu::SubgridCompressibleEulerSolver::float_type const* volumes);

  /// @brief This kernel computes the kepes flux at every outer faces
  ///        between neighboring grid elements and adds every
  ///        contribution to the fluxes variables. The flux variable
  ///        must be set beforehand to zeros.
  ///
  /// @param [in]  connectivity    face connectivity information.
  /// @param [in] level_difference level difference between
  ///                              neighboring element of a face. The
  ///                              mesh must be balanced so the level
  ///                              difference cannot exceed 1 in
  ///                              absolute value. Moreover, the
  ///                              assumption is made that the level
  ///                              of the left neighbor is greater or
  ///                              equal to the level of the right
  ///                              neighbor of a face.
  /// @param [in] neighbor_offset  the offset of the subelement in the
  ///                              right neighbor element of the face
  ///                              that matches with the left face.
  /// @param [in]  variables variables used to compute the fluxes.
  /// @param [out] fluxes    per subgrid element fluxes variables.
  __global__ void compute_outer_fluxes(t8gpu::SubgridMeshConnectivityAccessor<t8gpu::SubgridCompressibleEulerSolver::float_type, t8gpu::SubgridCompressibleEulerSolver::subgrid_type> connectivity,
				       t8_locidx_t const* level_difference,
				       t8_locidx_t const* neighbor_offset,
				       t8gpu::SubgridMemoryAccessorAll<t8gpu::VariableList, t8gpu::SubgridCompressibleEulerSolver::subgrid_type> variables,
				       t8gpu::SubgridMemoryAccessorAll<t8gpu::VariableList, t8gpu::SubgridCompressibleEulerSolver::subgrid_type> fluxes);

  /// @brief This kernel computes the refinement criteria per element.
  ///
  /// @param [in] variables            variables used to compute
  ///                                  the refinement criteria.
  /// @param [out] refinement_criteria the refinement criteria buffer to be filled.
  /// @param [in]  volume              volumes of each elements.
  /// @param [in]  num_local_elements  the number of local elements.
  __global__ void compute_refinement_criteria(typename t8gpu::SubgridCompressibleEulerSolver::subgrid_type::Accessor<t8gpu::SubgridCompressibleEulerSolver::float_type> density,
					      t8gpu::SubgridCompressibleEulerSolver::float_type* refinement_criteria,
					      t8gpu::SubgridCompressibleEulerSolver::float_type const* volumes,
					      t8_locidx_t num_local_elements);

}  // namespace t8gpu

#endif  // EXAMPLES_RAYLEIGH_TAYLOR_KERNELS_H
