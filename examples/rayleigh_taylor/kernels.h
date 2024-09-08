#ifndef EXAMPLES_RAYLEIGH_TAYLOR_KERNELS_H
#define EXAMPLES_RAYLEIGH_TAYLOR_KERNELS_H

#include <t8gpu/memory/subgrid_memory_manager.h>
#include <t8gpu/mesh/subgrid_mesh_manager.h>

#include "solver.h"

namespace t8gpu {

  __global__ void init_variable(typename t8gpu::SubgridCompressibleEulerSolver::subgrid_type::Accessor<t8gpu::SubgridCompressibleEulerSolver::float_type> variables);

  __global__ void compute_inner_fluxes(typename t8gpu::SubgridCompressibleEulerSolver::subgrid_type::Accessor<t8gpu::SubgridCompressibleEulerSolver::float_type> density,
				       typename t8gpu::SubgridCompressibleEulerSolver::subgrid_type::Accessor<t8gpu::SubgridCompressibleEulerSolver::float_type> fluxes,
				       t8gpu::SubgridCompressibleEulerSolver::float_type const* volumes);

  __global__ void compute_outer_fluxes(t8gpu::SubgridMeshConnectivityAccessor<t8gpu::SubgridCompressibleEulerSolver::float_type, t8gpu::SubgridCompressibleEulerSolver::subgrid_type> connectivity,
				       t8_locidx_t const* face_level_difference,
				       t8_locidx_t const* face_neighbor_offset,
				       t8gpu::SubgridMemoryAccessorAll<t8gpu::VariableList, t8gpu::SubgridCompressibleEulerSolver::subgrid_type> density,
				       t8gpu::SubgridMemoryAccessorAll<t8gpu::VariableList, t8gpu::SubgridCompressibleEulerSolver::subgrid_type> fluxes);

  __global__ void euler_update_density(t8gpu::SubgridCompressibleEulerSolver::subgrid_type::Accessor<t8gpu::SubgridCompressibleEulerSolver::float_type> density_prev,
				       t8gpu::SubgridCompressibleEulerSolver::subgrid_type::Accessor<t8gpu::SubgridCompressibleEulerSolver::float_type> density_next,
				       t8gpu::SubgridCompressibleEulerSolver::subgrid_type::Accessor<t8gpu::SubgridCompressibleEulerSolver::float_type> fluxes,
				       t8gpu::SubgridCompressibleEulerSolver::float_type const* volumes,
				       t8gpu::SubgridCompressibleEulerSolver::float_type delta_t);

}  // namespace t8gpu

#endif  // EXAMPLES_RAYLEIGH_TAYLOR_KERNELS_H
