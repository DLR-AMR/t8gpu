#ifndef SOLVERS_COMPRESSIBLE_EULER_KERNELS_H
#define SOLVERS_COMPRESSIBLE_EULER_KERNELS_H

#include <solvers/compressible_euler/solver.h>
#include <memory/memory_manager.h>
#include <mesh/mesh_manager.h>

namespace t8gpu {

__global__ void kepes_compute_fluxes(t8gpu::MemoryAccessorAll<t8gpu::VariableList> variables,
				     t8gpu::MemoryAccessorAll<t8gpu::VariableList> fluxes,
				     t8gpu::MeshConnectivityAccessor<typename t8gpu::variable_traits<t8gpu::VariableList>::float_type, 3> connectivity,
				     typename t8gpu::variable_traits<t8gpu::VariableList>::float_type* __restrict__ speed_estimates,
				     int num_faces);

}

#endif // SOLVERS_COMPRESSIBLE_EULER_KERNELS_H
