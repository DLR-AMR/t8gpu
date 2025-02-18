#include <cstdio>
#include <string>

#include "solver.h"

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
    size_t           dim    = 3;
    sc_MPI_Comm      comm   = MPI_COMM_WORLD;
    t8_scheme_cxx_t* scheme = t8_scheme_new_default_cxx();
    t8_cmesh_t       cmesh  = t8_cmesh_new_prismed_spherical_shell_icosahedron(0.8, 0.2, 2, 1, comm);
    t8_forest_t      forest = t8_forest_new_uniform(cmesh, scheme, 2, true, comm);

    t8gpu::CompressibleEulerSolver solver{comm, scheme, cmesh, forest};

    t8gpu::CompressibleEulerSolver::float_type delta_t = 0.0005f;

    for (size_t i = 0; i < 20'000; i++) {
      if (i % 100 == 0) {
        solver.adapt();
      }
      solver.iterate(delta_t);
      if (i % 100 == 0) {
        solver.save_conserved_variables_to_vtk("conserved_variables" + std::to_string(i / 100));
      }
    }
  }

  sc_finalize();

  mpiret = sc_MPI_Finalize();
  SC_CHECK_MPI(mpiret);

  return 0;
}
