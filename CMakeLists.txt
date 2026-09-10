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

# If t8code was built with MPI support, we need to find MPI and add its include directories
# This is necessary because nvcc does not automatically inherit include directories
# from linked targets like regular C/C++ compilers do
if (T8CODE_ENABLE_MPI)
  find_package(MPI REQUIRED)
  # MPI::MPI_C is already a dependency of T8CODE::T8, but nvcc needs explicit include paths
  # Use MPI_C_INCLUDE_PATH which is set by FindMPI.cmake
  if(MPI_C_INCLUDE_PATH)
    target_include_directories(t8gpu SYSTEM INTERFACE ${MPI_C_INCLUDE_PATH})
  elseif(MPI_C_INCLUDE_DIRS)
    target_include_directories(t8gpu SYSTEM INTERFACE ${MPI_C_INCLUDE_DIRS})
  elseif(TARGET MPI::MPI_C)
    get_target_property(MPI_INCLUDE_DIRS MPI::MPI_C INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(t8gpu SYSTEM INTERFACE ${MPI_INCLUDE_DIRS})
  else()
    # Last resort: use the include directory from MPI_C_COMPILER
    get_filename_component(MPI_INCLUDE_DIR ${MPI_C_COMPILER} DIRECTORY)
    get_filename_component(MPI_INCLUDE_DIR ${MPI_INCLUDE_DIR}/../include ABSOLUTE)
    target_include_directories(t8gpu SYSTEM INTERFACE ${MPI_INCLUDE_DIR})
  endif()
  
  # Set a variable for the MPI include directory that can be used by examples
  # Extract it from T8CODE::T8 target which already has MPI properly configured
  get_target_property(T8CODE_INCLUDE_DIRS T8CODE::T8 INTERFACE_INCLUDE_DIRECTORIES)
  # Find the MPI include directory in T8CODE's include directories
  # We look for the path containing "mpich" or "mpi" 
  foreach(_dir IN LISTS T8CODE_INCLUDE_DIRS)
    if(_dir MATCHES "mpich" OR _dir MATCHES "mpi")
      set(T8GPU_MPI_INCLUDE_DIR ${_dir})
      break()
    endif()
  endforeach()
  
  # Fail if we couldn't find the MPI include directory
  if(NOT T8GPU_MPI_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find MPI include directory from T8CODE::T8 target. "
      "Please ensure t8code was built with MPI support and the MPI include path is accessible.")
  endif()
endif()

target_link_libraries(t8gpu INTERFACE T8CODE::T8)

include(cmake/install.cmake)

if(T8GPU_BUILD_DOCUMENTATION)
  add_subdirectory(docs)
endif()

if(T8GPU_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
