add_executable(kelvin_helmholtz
  kernels.cu
  kernels.h
  main.cu
  solver.cu
  solver.h)

set_target_properties(kelvin_helmholtz PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(kelvin_helmholtz t8gpu)
