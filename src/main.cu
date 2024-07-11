#include <advection_solver.h>
#include <utils/profiling.h>

#include <cstdio>
#include <string>

int main(int argc, char* argv[]) {
  int mpiret = sc_MPI_Init(&argc, &argv);
  SC_CHECK_MPI(mpiret);

  sc_init(sc_MPI_COMM_WORLD, 1, 1, NULL, SC_LP_ESSENTIAL);
  t8_init(SC_LP_PRODUCTION);

  int rank, nb_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nb_ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  {
    t8gpu::AdvectionSolver advection_solver {};

    T8GPU_TIMER_START(iterations);

    size_t i = 0;
    double t = 0;
    t8gpu::AdvectionSolver::float_type delta_t = 0;
    double t_max = 0.75;
    for (; t < t_max; t += delta_t, i++) {
      if (rank == 0)
	std::cout << "t:" << t << ", delta_t:" << delta_t << std::endl;

      delta_t = advection_solver.compute_timestep();
      advection_solver.iterate(delta_t);
    }
    if (rank == 0)
      std::cout << "t:" << t << ", delta_t:" << delta_t << std::endl;

    T8GPU_TIMER_STOP(iterations);

    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), "advection_step_%05zu", i + 1);
    std::string prefix {buffer};
    advection_solver.save_vtk(prefix);
  }

  sc_finalize();

  mpiret = sc_MPI_Finalize();
  SC_CHECK_MPI(mpiret);

  return 0;
}
