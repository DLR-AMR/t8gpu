cmake_minimum_required(VERSION 3.16)

set(CMAKE_CUDA_ARCHITECTURES 86)

project(T8CODE_GPU_MINIAPP
  DESCRIPTION "A miniapp to test using t8code meshing capabilities in tandem with GPUs"
  LANGUAGES C CXX CUDA)

# module load PrgEnv/gcc11-openmpi-cuda
# cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../install -DT8CODE_ROOT=../../builds/install -DP4EST_ROOT=../../builds/install
find_package(P4EST)
find_package(T8CODE)

add_executable(miniapp)
target_sources(miniapp PRIVATE src/main.cu)

# target_compile_features(miniapp PRIVATE cxx_std_17)
# target_compile_options(miniapp PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(miniapp PRIVATE T8CODE::T8)
