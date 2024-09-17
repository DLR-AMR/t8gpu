add_executable(rayleigh_taylor_2d
  kernels_2d.cu
  kernels.inl
  kernels.h
  main_2d.cu
  solver_2d.cu
  solver.inl
  solver.h)

set_target_properties(rayleigh_taylor_2d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(rayleigh_taylor_2d t8gpu)

add_executable(rayleigh_taylor_3d
  kernels_3d.cu
  kernels.inl
  kernels.h
  main_3d.cu
  solver_3d.cu
  solver.inl
  solver.h)

set_target_properties(rayleigh_taylor_3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(rayleigh_taylor_3d t8gpu)
