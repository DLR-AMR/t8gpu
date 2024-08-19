add_executable(t8gpu)
target_sources(t8gpu PRIVATE
  main.cu
  memory/memory_manager.h
  memory/memory_manager.inl
  memory/shared_device_vector.h
  memory/shared_device_vector.inl
  mesh_manager.h
  mesh_manager.inl
  solvers/compressible_euler/kernels.cu
  solvers/compressible_euler/kernels.h
  solvers/compressible_euler/solver.cu
  solvers/compressible_euler/solver.h
  timestepping/ssp_runge_kutta.h
  timestepping/ssp_runge_kutta.inl
  utils/cuda.h
  utils/meta.h
  utils/profiling.h)
target_include_directories(t8gpu PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_compile_options(t8gpu PRIVATE -Wall -Wextra -Wno-unused-parameter -Wconversion --expt-relaxed-constexpr)

# embed kernel source code debug info in the binary
target_compile_options(t8gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)

set_target_properties(t8gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(T8CODE REQUIRED)
target_link_libraries(t8gpu PRIVATE T8CODE::T8)
