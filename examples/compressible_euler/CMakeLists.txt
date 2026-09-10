add_executable(kelvin_helmholtz
  kernels.cu
  kernels.h
  main.cu
  solver.cu
  solver.h)

set_target_properties(kelvin_helmholtz PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# nvcc doesn't support compiler flags like -Werror, -Wextra, -MD
# t8code passes these through -DT8_CXXFLAGS="... -Werror -Wextra ..."
# We need to override these definitions for CUDA files
# Use target_compile_definitions to override the inherited definitions
target_compile_definitions(kelvin_helmholtz PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    T8_CFLAGS="-pedantic -g -O3 -DNDEBUG -Wall"
    T8_CXXFLAGS="-pedantic -g -O3 -DNDEBUG -Wall"
    T8_LDFLAGS=""
  >)

# Add CUDA-safe compiler flags
target_compile_options(kelvin_helmholtz PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler -Wall,-Wno-unused-parameter,-Wconversion
    --generate-line-info
    --expt-relaxed-constexpr
  >)

target_link_libraries(kelvin_helmholtz t8gpu)
