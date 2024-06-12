cmake_minimum_required(VERSION 3.16)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(T8CODE_GPU_MINIAPP
  DESCRIPTION "A miniapp to test using t8code meshing capabilities in tandem with GPUs"
  LANGUAGES C CXX CUDA)

# module load PrgEnv/gcc11-openmpi-cuda
# cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../install -DT8CODE_ROOT=../../builds/install -DP4EST_ROOT=../../builds/install
find_package(P4EST)
find_package(T8CODE)

add_executable(miniapp)
target_sources(miniapp PRIVATE
  src/main.cu
  src/advection_solver.cu
  src/utils/cuda.h
  src/utils/profiling.h)
target_include_directories(miniapp PRIVATE src)

target_compile_options(miniapp PRIVATE -Wall -Wextra -Wno-unused-parameter)

# embed kernel source code debug info in the binary
target_compile_options(miniapp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
set_property(TARGET miniapp PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(miniapp PRIVATE T8CODE::T8)

