#include <shared_device_vector.h>

/// default implementation
template<typename T>
inline t8gpu::SharedDeviceVector<T>::SharedDeviceVector(size_t size, sc_MPI_Comm comm) : size_(size), comm_(comm), capacity_(size) {
  MPI_Comm_size(comm_, &nb_ranks_);
  MPI_Comm_rank(comm_, &rank_);

  handles_.resize(nb_ranks_);
  arrays_.resize(nb_ranks_);
  thrust::fill(arrays_.begin(), arrays_.end(), nullptr);

  if (size > 0) {
    T8GPU_CUDA_CHECK_ERROR(cudaMalloc(&arrays_[rank_], sizeof(T)*size_));
    T8GPU_CUDA_CHECK_ERROR(cudaIpcGetMemHandle(&(handles_[rank_].handle), arrays_[rank_]));
    handles_[rank_].need_open_ipc = true;
  } else {
    handles_[rank_].need_open_ipc = false;
  }
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles_.data(), sizeof(SharedDeviceVector<T>::handle_t), MPI_BYTE, comm_);

  for (int i=0; i<nb_ranks_; i++) {
    if (i != rank_ && handles_[i].need_open_ipc) {
      T8GPU_CUDA_CHECK_ERROR(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&arrays_[i]), handles_[i].handle, cudaIpcMemLazyEnablePeerAccess));
    }
  }
  device_arrays_ = arrays_;
}

template<typename T>
inline t8gpu::SharedDeviceVector<T>::~SharedDeviceVector() {
  for (int i=0; i<nb_ranks_; i++) {
    if (arrays_[i] != nullptr) {
      if (i == rank_) {
	T8GPU_CUDA_CHECK_ERROR(cudaFree(arrays_[i]));
      } else {
	T8GPU_CUDA_CHECK_ERROR(cudaIpcCloseMemHandle(arrays_[i]));
      }
    }
  }
}

template<typename T>
inline t8gpu::SharedDeviceVector<T>::SharedDeviceVector(SharedDeviceVector<T>&& other) : rank_(other.rank_),
											       nb_ranks_(other.nb_ranks_),
											       size_(other.size_),
											       capacity_(other.capacity_),
											       handles_(std::move(other.handles_)),
											       arrays_(other.arrays_),
											       device_arrays_(std::move(other.device_arrays_)) {
  for (int i=0; i<nb_ranks_; i++) {
    other.arrays_[i] = nullptr;
  }
}

template<typename T>
inline t8gpu::SharedDeviceVector<T>& t8gpu::SharedDeviceVector<T>::operator=(SharedDeviceVector&& other) {
  this->~SharedDeviceVector();
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
}

template<typename T>
inline void t8gpu::SharedDeviceVector<T>::resize(size_t new_size) {
  if (new_size <= capacity_) {
    size_ = new_size;
    handles_[rank_].need_open_ipc = false;
  } else {
    T8GPU_CUDA_CHECK_ERROR(cudaFree(arrays_[rank_]));

    capacity_ = new_size + new_size / 2;
    size_ = new_size;

    T* new_allocation {};
    T8GPU_CUDA_CHECK_ERROR(cudaMalloc(&new_allocation, sizeof(T)*capacity_));
    arrays_[rank_] = new_allocation;

    T8GPU_CUDA_CHECK_ERROR(cudaIpcGetMemHandle(&(handles_[rank_].handle), arrays_[rank_]));
    handles_[rank_].need_open_ipc = true;
  }
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles_.data(), sizeof(SharedDeviceVector<T>::handle_t), MPI_BYTE, comm_);
  for (int i=0; i<nb_ranks_; i++) {
    if (i != rank_ && handles_[i].need_open_ipc) {
      if (arrays_[i] != nullptr) {
	T8GPU_CUDA_CHECK_ERROR(cudaIpcCloseMemHandle(arrays_[i]));
      }
      T8GPU_CUDA_CHECK_ERROR(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&arrays_[i]), handles_[i].handle, cudaIpcMemLazyEnablePeerAccess));
    }
  }
  device_arrays_ = arrays_;
}

template<typename T>
inline const t8gpu::SharedDeviceVector<T>& t8gpu::SharedDeviceVector<T>::operator=(const thrust::host_vector<T>& other) {
  this->resize(other.size());
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(arrays_[rank_], thrust::raw_pointer_cast(other.data()), sizeof(T)*size_, cudaMemcpyHostToDevice));
  return *this;
}

template<typename T>
inline const t8gpu::SharedDeviceVector<T>& t8gpu::SharedDeviceVector<T>::operator=(const thrust::device_vector<T>& other) {
  this->resize(other.size());
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(arrays_[rank_], thrust::raw_pointer_cast(other.data()), sizeof(T)*size_, cudaMemcpyDeviceToDevice));
  return *this;
}
template<typename T>
[[nodiscard]] inline size_t t8gpu::SharedDeviceVector<T>::size() const { return size_; }

template<typename T>
inline void t8gpu::SharedDeviceVector<T>::clear() { size_ = 0; }

template<typename T>
[[nodiscard]] inline T* t8gpu::SharedDeviceVector<T>::get_own() { return arrays_[rank_]; }

template<typename T>
[[nodiscard]] inline T** t8gpu::SharedDeviceVector<T>::get_all() { return thrust::raw_pointer_cast(device_arrays_.data()); }

template<typename T>
[[nodiscard]] inline T const* t8gpu::SharedDeviceVector<T>::get_own() const { return arrays_[rank_]; }

template<typename T>
[[nodiscard]] inline T const* const* t8gpu::SharedDeviceVector<T>::get_all() const { return thrust::raw_pointer_cast(device_arrays_.data()); }

// Specialization for std::array<T, N>
template<typename T, size_t N>
inline t8gpu::SharedDeviceVector<std::array<T, N>>::SharedDeviceVector(size_t size, sc_MPI_Comm comm) : size_(size), comm_(comm), capacity_(size) {
  MPI_Comm_size(comm_, &nb_ranks_);
  MPI_Comm_rank(comm_, &rank_);

  handles_.resize(nb_ranks_);
  allocations_.resize(nb_ranks_);
  arrays_.resize(nb_ranks_ * N);
  thrust::fill(allocations_.begin(), allocations_.end(), nullptr);
  thrust::fill(arrays_.begin(), arrays_.end(), nullptr);

  if (size > 0) {
    T8GPU_CUDA_CHECK_ERROR(cudaMalloc(&allocations_[rank_], sizeof(T)*capacity_* N));
    T8GPU_CUDA_CHECK_ERROR(cudaIpcGetMemHandle(&(handles_[rank_].handle), allocations_[rank_]));
    handles_[rank_].need_open_ipc = true;
    handles_[rank_].capacity = capacity_;
  } else {
    handles_[rank_].need_open_ipc = false;
    handles_[rank_].capacity = capacity_;
  }
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles_.data(), sizeof(SharedDeviceVector<std::array<T, N>>::handle_t), MPI_BYTE, comm_);

  for (int i=0; i<nb_ranks_; i++) {
    if (i != rank_ && handles_[i].need_open_ipc) {
      T8GPU_CUDA_CHECK_ERROR(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&allocations_[i]), handles_[i].handle, cudaIpcMemLazyEnablePeerAccess));
    }
  }
  for (size_t i=0; i<N; i++) {
    for (int j=0; j<nb_ranks_; j++) {
      arrays_[i*nb_ranks_+j] = allocations_[j] + i*handles_[j].capacity;
    }
  }
  device_arrays_ = arrays_;
}

template<typename T, size_t N>
inline t8gpu::SharedDeviceVector<std::array<T, N>>::~SharedDeviceVector() {
  for (int i=0; i<nb_ranks_; i++) {
    if (allocations_[i] != nullptr) {
      if (i == rank_) {
	T8GPU_CUDA_CHECK_ERROR(cudaFree(allocations_[i]));
      } else {
	T8GPU_CUDA_CHECK_ERROR(cudaIpcCloseMemHandle(allocations_[i]));
      }
    }
  }
}

template<typename T, size_t N>
inline t8gpu::SharedDeviceVector<std::array<T, N>>::SharedDeviceVector(SharedDeviceVector<std::array<T, N>>&& other) : rank_(other.rank_),
														       nb_ranks_(other.nb_ranks_),
														       size_(other.size_),
														       capacity_(other.capacity_),
														       handles_(std::move(other.handles_)),
														       allocations_(other.allocations_),
														       arrays_(other.arrays_),

														       device_arrays_(std::move(other.device_arrays_)) {
  for (int i=0; i<nb_ranks_; i++) {
    other.allocations_[i] = nullptr;
  }
}

template<typename T, size_t N>
inline t8gpu::SharedDeviceVector<std::array<T, N>>& t8gpu::SharedDeviceVector<std::array<T, N>>::operator=(SharedDeviceVector&& other) {
  this->~SharedDeviceVector();
  rank_ = other.rank_;
  nb_ranks_ = other.nb_ranks_;
  size_ = other.size_;
  capacity_ = other.capacity_;
  handles_ = std::move(other.handles_);
  arrays_ = other.arrays_;
  allocations_ = other.allocations_;
  device_arrays_ = std::move(other.device_arrays_);

  for (int i=0; i<nb_ranks_; i++) {
    other.allocations_[i] = nullptr;
  }
  return *this;
}

template<typename T, size_t N>
inline void t8gpu::SharedDeviceVector<std::array<T, N>>::resize(size_t new_size) {
  if (new_size <= capacity_) {
    size_ = new_size;
    handles_[rank_].need_open_ipc = false;
  } else {
    T8GPU_CUDA_CHECK_ERROR(cudaFree(allocations_[rank_]));

    capacity_ = new_size + new_size / 2;
    size_ = new_size;

    T* new_allocation {};
    T8GPU_CUDA_CHECK_ERROR(cudaMalloc(&new_allocation, sizeof(T)*capacity_ * N));
    allocations_[rank_] = new_allocation;
    for (size_t j=0; j<N; j++) {
      arrays_[j*nb_ranks_+rank_] = allocations_[rank_] + j*capacity_;
    }

    T8GPU_CUDA_CHECK_ERROR(cudaIpcGetMemHandle(&(handles_[rank_].handle), allocations_[rank_]));
    handles_[rank_].need_open_ipc = true;
    handles_[rank_].capacity = capacity_;
  }
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles_.data(), sizeof(SharedDeviceVector<std::array<T, N>>::handle_t), MPI_BYTE, comm_);
  for (int i=0; i<nb_ranks_; i++) {
    if (i != rank_ && handles_[i].need_open_ipc) {
      if (allocations_[i] != nullptr) {
	T8GPU_CUDA_CHECK_ERROR(cudaIpcCloseMemHandle(allocations_[i]));
      }
      T8GPU_CUDA_CHECK_ERROR(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&allocations_[i]), handles_[i].handle, cudaIpcMemLazyEnablePeerAccess));
      for (size_t j=0; j<N; j++) {
	arrays_[j*nb_ranks_+i] = allocations_[i] + j*handles_[i].capacity;
      }
    }
  }
  device_arrays_ = arrays_;
}

template<typename T, size_t N>
inline void t8gpu::SharedDeviceVector<std::array<T, N>>::copy(int index, const thrust::host_vector<T>& other) {
  this->resize(other.size());
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(arrays_[index*nb_ranks_+rank_], thrust::raw_pointer_cast(other.data()), sizeof(T)*size_, cudaMemcpyHostToDevice));
}

template<typename T, size_t N>
inline void t8gpu::SharedDeviceVector<std::array<T, N>>::copy(int index, const thrust::device_vector<T>& other) {
  this->resize(other.size());
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(arrays_[index*nb_ranks_+rank_], thrust::raw_pointer_cast(other.data()), sizeof(T)*size_, cudaMemcpyDeviceToDevice));
}
template<typename T, size_t N>
[[nodiscard]] inline size_t t8gpu::SharedDeviceVector<std::array<T, N>>::size() const { return size_; }

template<typename T, size_t N>
inline void t8gpu::SharedDeviceVector<std::array<T, N>>::clear() { size_ = 0; }

template<typename T, size_t N>
[[nodiscard]] inline T* t8gpu::SharedDeviceVector<std::array<T, N>>::get_own(int index) { return arrays_[index*nb_ranks_+rank_]; }

template<typename T, size_t N>
[[nodiscard]] inline T** t8gpu::SharedDeviceVector<std::array<T, N>>::get_all(int index) { return thrust::raw_pointer_cast(device_arrays_.data() + index*nb_ranks_); }

template<typename T, size_t N>
[[nodiscard]] inline T const* t8gpu::SharedDeviceVector<std::array<T, N>>::get_own(int index) const { return arrays_[index*nb_ranks_+rank_]; }

template<typename T, size_t N>
[[nodiscard]] inline T const* const* t8gpu::SharedDeviceVector<std::array<T, N>>::get_all(int index) const { return thrust::raw_pointer_cast(device_arrays_.data() + index*nb_ranks_); }
