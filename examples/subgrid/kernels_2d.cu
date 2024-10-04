#define KERNELS_INCLUDE_IMPLEMENTATION
#include "kernels.h"

using SubgridType = t8gpu::Subgrid<4, 4>;

// We explicitely instantiate the kernel functions for the 2D subgrid
// in a separate compilation unit for better compilation speed
// (instead of having one big translation unit, we split it up in 3:
// main_2d.cu, solver_2d.cu and kernels_2d.cu).
template __global__ void compute_inner_fluxes<SubgridType>(
    t8gpu::SubgridMemoryAccessorOwn<VariableList, SubgridType>              variables,
    t8gpu::SubgridMemoryAccessorOwn<VariableList, SubgridType>              fluxes,
    typename SubgridCompressibleEulerSolver<SubgridType>::float_type const* volumes);

template __global__ void compute_outer_fluxes<SubgridType>(
    typename t8gpu::SubgridMeshConnectivityAccessor<typename SubgridCompressibleEulerSolver<SubgridType>::float_type,
                                                    SubgridType> connectivity,
    t8gpu::SubgridMemoryAccessorAll<VariableList, SubgridType>   variables,
    t8gpu::SubgridMemoryAccessorAll<VariableList, SubgridType>   fluxes);

template __global__ void compute_boundary_fluxes<SubgridType>(
    typename t8gpu::SubgridMeshConnectivityAccessor<typename SubgridCompressibleEulerSolver<SubgridType>::float_type,
                                                    SubgridType> connectivity,
    t8gpu::SubgridMemoryAccessorOwn<VariableList, SubgridType>   variables,
    t8gpu::SubgridMemoryAccessorOwn<VariableList, SubgridType>   fluxes);

template __global__ void compute_refinement_criteria<SubgridType>(
    typename SubgridType::Accessor<typename SubgridCompressibleEulerSolver<SubgridType>::float_type> density,
    typename SubgridCompressibleEulerSolver<SubgridType>::float_type*       refinement_criteria,
    typename SubgridCompressibleEulerSolver<SubgridType>::float_type const* volumes,
    t8_locidx_t                                                             num_local_elements);
