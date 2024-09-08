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
    using float_type = double;

    float_type delta_t = 0.125 / 2.0;

    sc_MPI_Comm comm = MPI_COMM_WORLD;
    t8_scheme_cxx_t* scheme = t8_scheme_new_default_cxx();
    t8_cmesh_t cmesh = t8_cmesh_new_periodic(comm, 3);
    t8_forest_t forest = t8_forest_new_uniform(cmesh, scheme, 1, true, comm);

    SubgridCompressibleEulerSolver solver {comm, scheme, cmesh, forest};

    solver.adapt();

    for(int it=0; it<25; it++) {
      solver.save_conserved_variables_to_vtk("density" + std::to_string(it));

      solver.iterate(delta_t);
    }
  }

  sc_finalize();

  mpiret = sc_MPI_Finalize();
  SC_CHECK_MPI(mpiret);

  return 0;
}
