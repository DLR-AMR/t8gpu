add_executable(subgrid_2d
  kernels_2d.cu
  kernels.inl
  kernels.h
  main_2d.cu
  solver_2d.cu
  solver.inl
  solver.h)

set_target_properties(subgrid_2d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(subgrid_2d t8gpu)

add_executable(subgrid_3d
  kernels_3d.cu
  kernels.inl
  kernels.h
  main_3d.cu
  solver_3d.cu
  solver.inl
  solver.h)

set_target_properties(subgrid_3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(subgrid_3d t8gpu)

# If t8code uses MPI, we need to ensure MPI include directories are available to CUDA
# This is because nvcc does not automatically inherit include directories from linked targets
# Use the T8GPU_MPI_INCLUDE_DIR variable set in the main CMakeLists.txt
if (T8CODE_ENABLE_MPI AND T8GPU_MPI_INCLUDE_DIR)
  target_include_directories(subgrid_2d SYSTEM PRIVATE ${T8GPU_MPI_INCLUDE_DIR})
  target_include_directories(subgrid_3d SYSTEM PRIVATE ${T8GPU_MPI_INCLUDE_DIR})
endif()
