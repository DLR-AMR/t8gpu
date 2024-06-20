#ifndef SHARED_DEVICE_VECTOR_H
#define SHARED_DEVICE_VECTOR_H

#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <utility>
#include <utils/cuda.h>

template<typename T>
class shared_device_vector {
 public:
  inline shared_device_vector(size_t size = 0) : size_(size), capacity_(size) {
    MPI_Comm_size(MPI_COMM_WORLD, &nb_ranks_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

    handles_.resize(nb_ranks_);
    arrays_.resize(nb_ranks_);
    thrust::fill(arrays_.begin(), arrays_.end(), nullptr);

    if (size > 0) {
      CUDA_CHECK_ERROR(cudaMalloc(&arrays_[rank_], sizeof(T)*size_));
      CUDA_CHECK_ERROR(cudaIpcGetMemHandle(&(handles_[rank_].handle), arrays_[rank_]));
      handles_[rank_].need_open_ipc = true;
    } else {
      handles_[rank_].need_open_ipc = false;
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles_.data(), sizeof(shared_device_vector<T>::handle_t), MPI_BYTE, MPI_COMM_WORLD);

    for (int i=0; i<nb_ranks_; i++) {
      if (i != rank_ && handles_[i].need_open_ipc) {
	CUDA_CHECK_ERROR(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&arrays_[i]), handles_[i].handle, cudaIpcMemLazyEnablePeerAccess));
      }
    }
    device_arrays_ = arrays_;
  }

  inline shared_device_vector(const shared_device_vector& other) = delete;
  inline shared_device_vector(shared_device_vector&& other) : rank_(other.rank_),
							      nb_ranks_(other.nb_ranks_),
							      size_(other.size_),
							      capacity_(other.capacity_),
							      handles_(std::move(other.handles_)),
							      arrays_(other.arrays_),
							      device_arrays_(std::move(other.device_arrays_)) {
    for (int i=0; i<nb_ranks_; i++) {
      other.arrays_[i] = nullptr;
    }
  };

  inline shared_device_vector& operator=(const shared_device_vector& other) = delete;
  inline shared_device_vector& operator=(shared_device_vector&& other) {
    this->~shared_device_vector();
    rank_ = other.rank_;
    nb_ranks_ = other.nb_ranks_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    handles_ = std::move(other.handles_);
    arrays_ = other.arrays_;
    device_arrays_ = std::move(other.device_arrays_);

    for (int i=0; i<nb_ranks_; i++) {
      other.arrays_[i] = nullptr;
    }
    return *this;
  };

  inline ~shared_device_vector() {
    for (int i=0; i<nb_ranks_; i++) {
      if (arrays_[i] != nullptr) {
	if (i == rank_) {
	  CUDA_CHECK_ERROR(cudaFree(arrays_[i]));
	} else {
	  CUDA_CHECK_ERROR(cudaIpcCloseMemHandle(arrays_[i]));
	}
      }
    }
  }

  inline void resize(size_t new_size) {
    if (new_size <= capacity_) {
      size_ = new_size;
      handles_[rank_].need_open_ipc = false;
    } else {
      capacity_ = new_size + new_size / 2;

      T* new_allocation;
      CUDA_CHECK_ERROR(cudaMalloc(&new_allocation, sizeof(T)*capacity_));
      CUDA_CHECK_ERROR(cudaMemcpy(new_allocation, arrays_[rank_], sizeof(T)*size_, cudaMemcpyDeviceToDevice));
      size_ = new_size;

      CUDA_CHECK_ERROR(cudaFree(arrays_[rank_]));
      arrays_[rank_] = new_allocation;

      CUDA_CHECK_ERROR(cudaIpcGetMemHandle(&(handles_[rank_].handle), arrays_[rank_]));
      handles_[rank_].need_open_ipc = true;
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles_.data(), sizeof(shared_device_vector<T>::handle_t), MPI_BYTE, MPI_COMM_WORLD);
    for (int i=0; i<nb_ranks_; i++) {
      if (i != rank_ && handles_[i].need_open_ipc) {
	if (arrays_[i] != nullptr) {
	  CUDA_CHECK_ERROR(cudaIpcCloseMemHandle(arrays_[i]));
	}
	CUDA_CHECK_ERROR(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&arrays_[i]), handles_[i].handle, cudaIpcMemLazyEnablePeerAccess));
      }
    }
    device_arrays_ = arrays_;
  }

  inline const shared_device_vector<T>& operator=(const thrust::host_vector<T>& other) {
    this->resize(other.size());
    CUDA_CHECK_ERROR(cudaMemcpy(arrays_[rank_], thrust::raw_pointer_cast(other.data()), sizeof(T)*size_, cudaMemcpyHostToDevice));
    return *this;
  }

  inline const shared_device_vector<T>& operator=(const thrust::device_vector<T>& other) {
    this->resize(other.size());
    CUDA_CHECK_ERROR(cudaMemcpy(arrays_[rank_], thrust::raw_pointer_cast(other.data()), sizeof(T)*size_, cudaMemcpyDeviceToDevice));
    return *this;
  }

  inline size_t size() const { return size_; }

  inline void clear() {
    size_ = 0;
  }

  inline T* get_own() { return arrays_[rank_]; }
  inline T** get_all() { return thrust::raw_pointer_cast(device_arrays_.data()); }

  inline T const * get_own() const { return arrays_[rank_]; }
  inline T* const * get_all() const { return thrust::raw_pointer_cast(device_arrays_.data()); }

  static inline void swap(shared_device_vector<T>& a, shared_device_vector<T>& b) {
    std::swap(a.size_, b.size_);
    std::swap(a.capacity_, b.capacity_);
    std::swap(a.handles_, b.handles_);
    std::swap(a.arrays_, b.arrays_);
    std::swap(a.device_arrays_, b.device_arrays_);
  }

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
void std::swap(shared_device_vector<T>& a, shared_device_vector<T>& b) {
  shared_device_vector<T>::swap(a, b);
}

#endif // SHARED_DEVICE_VECTOR_H
