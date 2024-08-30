cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(T8GPU
  DESCRIPTION  "A library to test t8code meshing capabilities with GPUs"
  HOMEPAGE_URL "https://github.com/DLR-AMR/t8gpu"
  LANGUAGES    C CXX CUDA)

option(T8GPU_BUILD_DOCUMENTATION "Build t8gpu developer documentation" OFF)
option(T8GPU_BUILD_EXAMPLES      "Build t8gpu examples"                 ON)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_library(t8gpu INTERFACE)

target_include_directories(t8gpu
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

target_compile_options(t8gpu
  INTERFACE
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wconversion
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
    --expt-relaxed-constexpr)

find_package(T8CODE REQUIRED)
target_link_libraries(t8gpu INTERFACE T8CODE::T8)

include(cmake/install.cmake)

if(T8GPU_BUILD_DOCUMENTATION)
  add_subdirectory(docs)
endif()

if(T8GPU_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
