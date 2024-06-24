/// @file shared_device_vector.h
/// @brief This header file declares the shared device vector class

#ifndef SHARED_DEVICE_VECTOR_H
#define SHARED_DEVICE_VECTOR_H

#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <utility>
#include <utils/cuda.h>

namespace t8gpu {
  ///
  /// @brief A class that implements a shared array This class creates
  ///
  /// A shared vetor of device memory by providing most of the
  /// `std::vector` interface. Each MPI rank handles its owm local gpu
  /// allocation. However, as all of these allocation are done on the
  /// same GPU, each rank can retrieve device pointers to all of the
  /// other rank's allocations throught IPC (Inter Process
  /// Communication) in order to be able to launch kernels accessing
  /// other MPI ranks's GPU device allocated memory. This is primarily
  /// used to handle the ghost layer: instead of allocating on the
  /// device space for the ghost layer, we directly access the other
  /// rank's device allocations. This implies a better use of the GPU
  /// memory but requires more synhronization between each ranks GPU
  /// kernel launches.
  template<typename T>
  class SharedDeviceVector {
  public:
    /// @brief Constructor of the shared device array
    ///
    /// @param [in]         size size of the GPU allocation
    /// @param [in]         comm MPI communicator used
    /// @return             instance object of the vector class
    inline SharedDeviceVector(size_t size = 0, sc_MPI_Comm comm = sc_MPI_COMM_WORLD);

    /// @brief Destrutor of the shared device vector class
    ///
    /// This destructor frees all the allocated device memory.
    inline ~SharedDeviceVector();

    /// @brief Move copy constructor
    ///
    /// Steal the GPU allocations of `other` and leaves other in a valid state as necessary
    inline SharedDeviceVector(SharedDeviceVector&& other);
    inline SharedDeviceVector(const SharedDeviceVector& other) = delete;

    /// @brief Move assignment operator
    ///
    /// Steal the GPU allocations of `other` and leaves other in a valid state as necessary
    inline SharedDeviceVector& operator=(SharedDeviceVector&& other);
    inline SharedDeviceVector& operator=(const SharedDeviceVector& other) = delete;

    /// @brief Resizes the allocation to `new_size`
    /// @param [in]         new_size new size of the vector
    ///
    /// It is important to state that this function needs to be called
    /// by all MPI ranks. Otherwise this results in a lock. If a ranks
    /// does not need to change the size of the allocation, use resize
    /// with the current size of the vector. This functions may not
    /// need to reallocate even if new_size > size the copy over as
    /// the capacity of the allocation might be greater than new_size
    /// and if not we overallocate by a factor 1/2 to monimize
    /// reallocation on later calls to the resize function.
    inline void resize(size_t new_size);

    /// @brief conversion operator
    ///
    /// This member function converts a thrust host array to a shared
    /// device vector. This function is expensive ás host to device
    /// memory copy operation is necessary.
    inline const SharedDeviceVector<T>& operator=(const thrust::host_vector<T>& other);

    /// @brief conversion operator
    ///
    /// This member function converts a thrust device array to a
    /// shared device vector. This function is expensive ás device to
    /// device memory copy operation is necessary.
    inline const SharedDeviceVector<T>& operator=(const thrust::device_vector<T>& other);

    /// @brief Returns the size of the allocation
    ///
    /// @return returns the size of the allocation.
    [[nodiscard]] inline size_t size() const;

    /// @brief Clears the content of the vector
    ///
    /// Clears the context of the vector but does not necessarily
    /// deallocates device GPU memory.
    inline void clear();

    /// @brief Return device allocated memory
    ///
    /// @return returns device allocated memory
    ///
    /// Returns a device pointer to the current device
    /// allocation. This pointer is invalided as soon as the resize
    /// member function is invoked as reallocation might be necessary.
    [[nodiscard]] inline T* get_own();

    /// @brief Return device allocated memory of all the MPI ranks
    ///
    /// @return returns device allocated memory of all the MPI ranks
    ///
    /// Returns a pointer to device memory containing an array of
    /// length the number of MPI ranks, with each element being
    /// pointers to device memory accessible from the current MPI rank
    /// of the other rank's device allocation. This pointer is
    /// invalided as soon as the resize member function is invoked as
    /// reallocation might be necessary.
    [[nodiscard]] inline T** get_all();

    /// @brief Return device allocated memory
    ///
    /// @return returns device allocated memory
    ///
    /// Returns a device pointer to the current device
    /// allocation. This pointer is invalided as soon as the resize
    /// member function is invoked as reallocation might be necessary.
    [[nodiscard]] inline T const* get_own() const;

    /// @brief Return device allocated memory of all the MPI ranks
    ///
    /// @return returns device allocated memory of all the MPI ranks
    ///
    /// Returns a pointer to device memory containing an array of
    /// length the number of MPI ranks, with each element being
    /// pointers to device memory accessible from the current MPI rank
    /// of the other rank's device allocation. This pointer is
    /// invalided as soon as the resize member function is invoked as
    /// reallocation might be necessary.
    [[nodiscard]] inline T const* const* get_all() const;

    /// @brief swap two shared device vectors
    ///
    /// swap two shared device vectors. Prefer using the
    /// specialization of the standard library swap function std::swap
    static inline void swap(SharedDeviceVector<T>& a, SharedDeviceVector<T>& b);

  private:
    sc_MPI_Comm comm_;
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

  /// @brief swap two shared device vectors
  template<typename T>
  void std::swap(SharedDeviceVector<T>& a, SharedDeviceVector<T>& b);
} // namespace t8gpu

#include <shared_device_vector.inl>

#endif // SHARED_DEVICE_VECTOR_H
