#ifndef SHARED_DEVICE_VECTOR_H
#define SHARED_DEVICE_VECTOR_H

#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <utility>
#include <utils/cuda.h>

namespace t8gpu {
  template<typename T>
  class shared_device_vector {
  public:
    inline shared_device_vector(size_t size = 0);
    inline ~shared_device_vector();

    inline shared_device_vector(const shared_device_vector& other) = delete;
    inline shared_device_vector(shared_device_vector&& other);

    inline shared_device_vector& operator=(const shared_device_vector& other) = delete;
    inline shared_device_vector& operator=(shared_device_vector&& other);

    inline void resize(size_t new_size);

    inline const shared_device_vector<T>& operator=(const thrust::host_vector<T>& other);
    inline const shared_device_vector<T>& operator=(const thrust::device_vector<T>& other);

    [[nodiscard]] inline size_t size() const;
    inline void clear();

    [[nodiscard]] inline T* get_own();
    [[nodiscard]] inline T** get_all();

    [[nodiscard]] inline T const* get_own() const;
    [[nodiscard]] inline T const* const* get_all() const;

    static inline void swap(shared_device_vector<T>& a, shared_device_vector<T>& b);

  private:
    int rank_;
    int nb_ranks_;
    size_t size_;
    size_t capacity_;

    struct handle_t {
      cudaIpcMemHandle_t handle;
      bool need_open_ipc;
    };
    thrust::host_vector<handle_t> handles_;
    thrust::host_vector<T*> arrays_;
    thrust::device_vector<T*> device_arrays_;
  };

  template<typename T>
  void std::swap(shared_device_vector<T>& a, shared_device_vector<T>& b);
} // namespace t8gpu

#include <shared_device_vector.inl>

#endif // SHARED_DEVICE_VECTOR_H
