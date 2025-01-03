#define SOLVER_INCLUDE_IMPLEMENTATION
#include "solver.h"

using SubgridType = t8gpu::Subgrid<16, 16>;

// We explicitely instantiate the solver type for the 2D subgrid in a
// separate compilation unit for better compilation speed (instead of
// having one big translation unit, we split it up in 3: main_2d.cu,
// solver_2d.cu and kernels_2d.cu).
template class SubgridCompressibleEulerSolver<SubgridType>;
