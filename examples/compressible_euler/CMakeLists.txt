add_executable(kelvin_helmholtz
  kernels.cu
  kernels.h
  main.cu
  solver.cu
  solver.h)

set_target_properties(kelvin_helmholtz PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(kelvin_helmholtz t8gpu)

# Link MPI library if t8code uses MPI
if (T8CODE_ENABLE_MPI)
  # Ensure MPI is found (it should be available from t8code)
  if(NOT TARGET MPI::MPI_C)
    find_package(MPI REQUIRED)
  endif()
endif()

# If t8code uses MPI, we need to ensure MPI include directories are available to CUDA
# This is because nvcc does not automatically inherit include directories from linked targets
# Explicitly add MPI include directories to the target
if (T8CODE_ENABLE_MPI)
  # First try to get the include directory from MPI_C_INCLUDE_PATH
  if(MPI_C_INCLUDE_PATH)
    target_include_directories(kelvin_helmholtz SYSTEM PRIVATE ${MPI_C_INCLUDE_PATH})
  elseif(TARGET MPI::MPI_C)
    # Try to get it from the MPI::MPI_C target
    get_target_property(MPI_INCLUDE_DIRS MPI::MPI_C INTERFACE_INCLUDE_DIRECTORIES)
    if(MPI_INCLUDE_DIRS)
      target_include_directories(kelvin_helmholtz SYSTEM PRIVATE ${MPI_INCLUDE_DIRS})
    else()
      # If that doesn't work, extract from MPI_C_COMPILER
      get_filename_component(MPI_INCLUDE_DIR ${MPI_C_COMPILER} DIRECTORY)
      get_filename_component(MPI_INCLUDE_DIR ${MPI_INCLUDE_DIR}/../include ABSOLUTE)
      target_include_directories(kelvin_helmholtz SYSTEM PRIVATE ${MPI_INCLUDE_DIR})
    endif()
  else()
    message(FATAL_ERROR "MPI::MPI_C target not found and MPI_C_INCLUDE_PATH not set. "
      "Please ensure MPI is properly configured.")
  endif()
endif()
