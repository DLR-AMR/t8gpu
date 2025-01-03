
#include <cstdio>
#include <string>

#include <t8.h>

#include "solver.h"

using namespace t8gpu;

int main(int argc, char* argv[]) {
  int mpiret = sc_MPI_Init(&argc, &argv);
  SC_CHECK_MPI(mpiret);

  sc_init(sc_MPI_COMM_WORLD, 1, 1, NULL, SC_LP_ESSENTIAL);
  t8_init(SC_LP_PRODUCTION);

  int rank, nb_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nb_ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // We create a dummy frame so that destructors of solver class are
  // called before MPI finalize.
  {
    using SubgridType = Subgrid<16, 16>;
    using float_type  = typename SubgridCompressibleEulerSolver<SubgridType>::float_type;

    float_type delta_t =
        static_cast<float_type>(0.05 * pow(0.5,
                                          SubgridMeshManager<VariableList, StepList, SubgridType>::max_level +
                                              t8gpu::meta::log2_v<SubgridType::template extent<0>>));

    sc_MPI_Comm comm   = MPI_COMM_WORLD;
    t8_scheme*  scheme = t8_scheme_new_default();
    t8_cmesh_t  cmesh  = t8_cmesh_new_periodic(comm, SubgridType::rank);
    t8_forest_t forest = t8_forest_new_uniform(cmesh, scheme, 8, true, comm);

    SubgridCompressibleEulerSolver<SubgridType> solver{comm, scheme, cmesh, forest};

    for (int it = 0; it < 4'000; it++) {
      if (it % 50 == 0) {
        solver.adapt();
      }

      std::cout << "it:" << it << std::endl;
      if (it % 10 == 0) {
        solver.save_density_to_vtk("density" + std::to_string(it / 10));
        solver.save_mesh_to_vtk("mesh" + std::to_string(it / 10));
      }

      solver.iterate(delta_t);
    }
  }

  sc_finalize();

  mpiret = sc_MPI_Finalize();
  SC_CHECK_MPI(mpiret);

  return 0;
}
