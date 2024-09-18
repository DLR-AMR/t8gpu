#ifndef EXAMPLES_SUBGRID_KERNELS_H
#define EXAMPLES_SUBGRID_KERNELS_H

#include <t8gpu/memory/subgrid_memory_manager.h>
#include <t8gpu/mesh/subgrid_mesh_manager.h>

#include "solver.h"

/// @brief This kernel computes the kepes flux at every inner faces
///        within subgrid elements and adds every contribution to
///        the fluxes variables. The flux variable must be
///        set beforehand to zeros.
///
/// @param [in]  variables variables used to compute the fluxes.
/// @param [out] fluxes    per subgrid element fluxes variables.
/// @param [in]  volume    volumes of each elements.
template<typename SubgridType>
__global__ void compute_inner_fluxes(t8gpu::SubgridMemoryAccessorOwn<VariableList, SubgridType>              variables,
                                     t8gpu::SubgridMemoryAccessorOwn<VariableList, SubgridType>              fluxes,
                                     typename SubgridCompressibleEulerSolver<SubgridType>::float_type const* volumes);

/// @brief This kernel computes the kepes flux at every outer faces
///        between neighboring grid elements and adds every
///        contribution to the fluxes variables. The flux variable
///        must be set beforehand to zeros.
///
/// @param [in]  connectivity    face connectivity information.
/// @param [in]  variables variables used to compute the fluxes.
/// @param [out] fluxes    per subgrid element fluxes variables.
template<typename SubgridType>
__global__ void compute_outer_fluxes(
    typename t8gpu::SubgridMeshConnectivityAccessor<typename SubgridCompressibleEulerSolver<SubgridType>::float_type,
                                                    SubgridType> connectivity,
    t8gpu::SubgridMemoryAccessorAll<VariableList, SubgridType>   variables,
    t8gpu::SubgridMemoryAccessorAll<VariableList, SubgridType>   fluxes);

/// @brief This kernel computes the refinement criteria per element.
///
/// @param [in] variables            variables used to compute
///                                  the refinement criteria.
/// @param [out] refinement_criteria the refinement criteria buffer to be filled.
/// @param [in]  volume              volumes of each elements.
/// @param [in]  num_local_elements  the number of local elements.
template<typename SubgridType>
__global__ void compute_refinement_criteria(
    typename SubgridType::Accessor<typename SubgridCompressibleEulerSolver<SubgridType>::float_type> density,
    typename SubgridCompressibleEulerSolver<SubgridType>::float_type*       refinement_criteria,
    typename SubgridCompressibleEulerSolver<SubgridType>::float_type const* volumes,
    t8_locidx_t                                                             num_local_elements);

#ifdef KERNELS_INCLUDE_IMPLEMENTATION
#include "kernels.inl"
#endif

#endif  // EXAMPLES_SUBGRID_KERNELS_H
