#include <advection_solver.h>
#include <utils/profiling.h>

#include <cstdio>
#include <string>

int main(int argc, char* argv[]) {
  int mpiret = sc_MPI_Init(&argc, &argv);
  SC_CHECK_MPI(mpiret);

  sc_init(sc_MPI_COMM_WORLD, 1, 1, NULL, SC_LP_ESSENTIAL);
  t8_init(SC_LP_PRODUCTION);

  {
    t8gpu::AdvectionSolver advection_solver {};

    size_t i = 0;
    for (; i < 10'000; i++) {
      if (i % 10 == 0) {
	advection_solver.adapt();
	advection_solver.partition();
	advection_solver.compute_connectivity_information();
      }
      advection_solver.iterate();

      if (i % 200 == 0) {
	char buffer[256];
	std::snprintf(buffer, sizeof(buffer), "advection_step_%05zu", i + 1);
	std::string prefix {buffer};
	advection_solver.save_vtk(prefix);
      }
    }
  }

  sc_finalize();

  mpiret = sc_MPI_Finalize();
  SC_CHECK_MPI(mpiret);

  return 0;
}
