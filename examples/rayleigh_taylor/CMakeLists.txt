add_executable(rayleigh_taylor
  # kernels.cu
  # kernels.h
  main.cu
  # solver.cu
  # solver.h
  )

set_target_properties(rayleigh_taylor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(rayleigh_taylor t8gpu)
