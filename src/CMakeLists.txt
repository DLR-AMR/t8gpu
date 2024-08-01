add_executable(t8gpu)
target_sources(t8gpu PRIVATE
  main.cu
  compressible_euler_solver.cu
  shared_device_vector.inl
  shared_device_vector.h
  timestepping/ssp_runge_kutta.inl
  timestepping/ssp_runge_kutta.h
  utils/cuda.h
  utils/meta.h
  utils/profiling.h)
target_include_directories(t8gpu PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_compile_options(t8gpu PRIVATE -Wall -Wextra -Wno-unused-parameter -Wconversion)

# embed kernel source code debug info in the binary
target_compile_options(t8gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
set_property(TARGET t8gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)

find_package(T8CODE REQUIRED)
target_link_libraries(t8gpu PRIVATE T8CODE::T8)
