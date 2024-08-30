#ifndef SOLVERS_COMPRESSIBLE_EULER_KERNELS_H
#define SOLVERS_COMPRESSIBLE_EULER_KERNELS_H

#include <t8gpu/memory/memory_manager.h>
#include <t8gpu/mesh/mesh_manager.h>

#include "solver.h"

namespace t8gpu {

  /// @brief This kernel computes the kepes flux at every faces and
  ///        adds every contribution to the fluxes variables. The flux
  ///        variable must be beforehand set to zeros.
  ///
  /// @param [in]  connectivity    face connectivity information.
  /// @param [in]  variables       all variables.
  /// @param [out] fluxes          flux contributions for each face of each
  ///                              element are added up in these variables.
  /// @param [out] speed_estimates wave speed estimates based on the HLL
  ///                              flux are written in this variable for
  ///                              each face.
  __global__ void kepes_compute_fluxes(t8gpu::MeshConnectivityAccessor<typename t8gpu::variable_traits<t8gpu::VariableList>::float_type, 3> connectivity,
				       t8gpu::MemoryAccessorAll<t8gpu::VariableList> variables,
				       t8gpu::MemoryAccessorAll<t8gpu::VariableList> fluxes,
				       typename t8gpu::variable_traits<t8gpu::VariableList>::float_type* __restrict__ speed_estimates);

  /// @brief This kernel computes the kepes flux at every boundary
  ///        faces using reflective boundary conditions and adds every
  ///        contribution to the fluxes variables.
  ///
  /// @param [in]  connectivity    face connectivity information.
  /// @param [in]  variables       owned variables.
  /// @param [out] fluxes          flux contributions for each face of each
  ///                              element are added up in these variables.
  /// @param [out] speed_estimates wave speed estimates based on the HLL
  ///                              flux are written in this variable for
  ///                              each face.
  __global__ void reflective_boundary_condition(t8gpu::MeshConnectivityAccessor<typename t8gpu::variable_traits<t8gpu::VariableList>::float_type, 3> connectivity,
						t8gpu::MemoryAccessorOwn<t8gpu::VariableList> variables,
						t8gpu::MemoryAccessorOwn<t8gpu::VariableList> fluxes,
						typename t8gpu::variable_traits<t8gpu::VariableList>::float_type* __restrict__ speed_estimates);

  /// @brief A simple density gradient estimator that can be used to
  ///        construct a refinement criterion. It estimates the
  ///        density gradient at each faces and sets the first flux
  ///        variable to a mean gradient between its adjacent faces.
  ///
  /// @param [in]  data_next   The conserved variables.
  /// @param [out] data_fluxes The first flux variable is where the
  ///                          gradient estimation will be computed.
  /// @param [in] connectivity Face connectivity infurmation.
  __global__ void estimate_gradient(t8gpu::MeshConnectivityAccessor<typename t8gpu::variable_traits<VariableList>::float_type, CompressibleEulerSolver::dim> connectivity,
				    t8gpu::MemoryAccessorAll<t8gpu::VariableList> data_next,
				    t8gpu::MemoryAccessorAll<t8gpu::VariableList> data_fluxes);

}

#endif // SOLVERS_COMPRESSIBLE_EULER_KERNELS_H
