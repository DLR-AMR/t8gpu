add_executable(subgrid_2d
  kernels_2d.cu
  kernels.inl
  kernels.h
  main_2d.cu
  solver_2d.cu
  solver.inl
  solver.h)

set_target_properties(subgrid_2d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# nvcc doesn't support compiler flags like -Werror, -Wextra, -MD
# t8code passes these through -DT8_CXXFLAGS="... -Werror -Wextra ..."
# Override the inherited definitions for CUDA files
target_compile_definitions(subgrid_2d PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    T8_CFLAGS="-pedantic -g -O3 -DNDEBUG -Wall"
    T8_CXXFLAGS="-pedantic -g -O3 -DNDEBUG -Wall"
    T8_LDFLAGS=""
  >)

# Add CUDA-safe compiler flags
target_compile_options(subgrid_2d PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler -Wall,-Wno-unused-parameter,-Wconversion
    --generate-line-info
    --expt-relaxed-constexpr
  >)
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
# nvcc doesn't support compiler flags like -Werror, -Wextra, -MD
# Override the inherited definitions for CUDA files
target_compile_definitions(subgrid_3d PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    T8_CFLAGS="-pedantic -g -O3 -DNDEBUG -Wall"
    T8_CXXFLAGS="-pedantic -g -O3 -DNDEBUG -Wall"
    T8_LDFLAGS=""
  >)

# Add CUDA-safe compiler flags
target_compile_options(subgrid_3d PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler -Wall,-Wno-unused-parameter,-Wconversion
    --generate-line-info
    --expt-relaxed-constexpr
  >)
target_link_libraries(subgrid_3d t8gpu)
