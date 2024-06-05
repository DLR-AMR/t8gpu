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

/* In debug mode, this macro serializes cuda calls as it synchronizes
 * the GPU in order to get any asynchronous errors during kernel
 * execution. In release mode, this macro does nothing */
#ifndef NDEBUG
#define CUDA_CHECK_LAST_ERROR() do {					\
    cudaDeviceSynchronize();						\
    cudaError_t value = cudaGetLastError();				\
    if (value != cudaSuccess) {						\
      std::cerr << "caught CUDA runtime error at: " __FILE__ ":" << __LINE__ << std::endl; \
      std::cerr << cudaGetErrorString(value) << std::endl;		\
      exit(EXIT_FAILURE);						\
    }									\
  } while(0)
#else
#define CUDA_CHECK_LAST_ERROR()
#endif // NDEBUG

#endif // CUDA_UTILS_H
