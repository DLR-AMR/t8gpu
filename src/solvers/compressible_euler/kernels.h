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

  __global__ void estimate_gradient(t8gpu::MemoryAccessorAll<t8gpu::VariableList> data_next,
				    t8gpu::MemoryAccessorAll<t8gpu::VariableList> data_fluxes,
				    typename t8gpu::variable_traits<t8gpu::VariableList>::float_type const* __restrict__ normal,
				    typename t8gpu::variable_traits<t8gpu::VariableList>::float_type const* __restrict__ area,
				    int const* e_idx, int* rank,
				    t8_locidx_t* indices, int nb_edges);

}

#endif // SOLVERS_COMPRESSIBLE_EULER_KERNELS_H
