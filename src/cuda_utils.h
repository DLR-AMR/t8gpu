#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cstdlib>

#define CUDA_CHECK_ERROR(expr) do {					\
    cudaError_t value = (expr);						\
    if (value != cudaSuccess) {						\
      std::cerr << "caught CUDA runtime error at: " __FILE__ ":" << __LINE__ << std::endl; \
      std::cerr << cudaGetErrorString(value) << std::endl;		\
      exit(EXIT_FAILURE);						\
    }									\
  } while(0)

#endif // CUDA_UTILS_H
