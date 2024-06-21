cmake_minimum_required(VERSION 3.16)

project(T8GPU
  DESCRIPTION "A miniapp to test using t8code meshing capabilities in tandem with GPUs"
  LANGUAGES C CXX CUDA)

option(T8GPU_BUILD_DOCUMENTATION "Build t8gpu developper documentation" ON)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src)

if(T8GPU_BUILD_DOCUMENTATION)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/docs)
endif()
