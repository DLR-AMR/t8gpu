add_executable(kelvin_helmholtz
  kernels.cu
  kernels.h
  main.cu
  solver.cu
  solver.h)

set_target_properties(kelvin_helmholtz PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(kelvin_helmholtz t8gpu)

# If t8code uses MPI, we need to ensure MPI include directories are available to CUDA
# This is because nvcc does not automatically inherit include directories from linked targets
# Use the T8GPU_MPI_INCLUDE_DIR variable set in the main CMakeLists.txt
if (T8CODE_ENABLE_MPI AND T8GPU_MPI_INCLUDE_DIR)
  target_include_directories(kelvin_helmholtz SYSTEM PRIVATE ${T8GPU_MPI_INCLUDE_DIR})
endif()
