add_executable(kelvin_helmholtz
  kernels.cu
  kernels.h
  main.cu
  solver.cu
  solver.h)

target_link_libraries(kelvin_helmholtz t8gpu)
