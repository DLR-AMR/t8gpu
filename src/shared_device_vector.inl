#include <shared_device_vector.h>

template<typename T>
inline t8gpu::SharedDeviceVector<T>::SharedDeviceVector(size_t size) : size_(size), capacity_(size) {
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
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles_.data(), sizeof(SharedDeviceVector<T>::handle_t), MPI_BYTE, MPI_COMM_WORLD);

  for (int i=0; i<nb_ranks_; i++) {
    if (i != rank_ && handles_[i].need_open_ipc) {
      CUDA_CHECK_ERROR(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&arrays_[i]), handles_[i].handle, cudaIpcMemLazyEnablePeerAccess));
    }
  }
  device_arrays_ = arrays_;
}

template<typename T>
inline t8gpu::SharedDeviceVector<T>::~SharedDeviceVector() {
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
    capacity_ = new_size + new_size / 2;

    T* new_allocation {};
    CUDA_CHECK_ERROR(cudaMalloc(&new_allocation, sizeof(T)*capacity_));
    CUDA_CHECK_ERROR(cudaMemcpy(new_allocation, arrays_[rank_], sizeof(T)*size_, cudaMemcpyDeviceToDevice));
    size_ = new_size;

    CUDA_CHECK_ERROR(cudaFree(arrays_[rank_]));
    arrays_[rank_] = new_allocation;

    CUDA_CHECK_ERROR(cudaIpcGetMemHandle(&(handles_[rank_].handle), arrays_[rank_]));
    handles_[rank_].need_open_ipc = true;
  }
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles_.data(), sizeof(SharedDeviceVector<T>::handle_t), MPI_BYTE, MPI_COMM_WORLD);
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

template<typename T>
inline const t8gpu::SharedDeviceVector<T>& t8gpu::SharedDeviceVector<T>::operator=(const thrust::host_vector<T>& other) {
  this->resize(other.size());
  CUDA_CHECK_ERROR(cudaMemcpy(arrays_[rank_], thrust::raw_pointer_cast(other.data()), sizeof(T)*size_, cudaMemcpyHostToDevice));
  return *this;
}

template<typename T>
inline const t8gpu::SharedDeviceVector<T>& t8gpu::SharedDeviceVector<T>::operator=(const thrust::device_vector<T>& other) {
  this->resize(other.size());
  CUDA_CHECK_ERROR(cudaMemcpy(arrays_[rank_], thrust::raw_pointer_cast(other.data()), sizeof(T)*size_, cudaMemcpyDeviceToDevice));
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

template<typename T>
inline void t8gpu::SharedDeviceVector<T>::swap(SharedDeviceVector<T>& a, SharedDeviceVector<T>& b) {
  std::swap(a.size_, b.size_);
  std::swap(a.capacity_, b.capacity_);
  std::swap(a.handles_, b.handles_);
  std::swap(a.arrays_, b.arrays_);
  std::swap(a.device_arrays_, b.device_arrays_);
}

template<typename T>
void std::swap(t8gpu::SharedDeviceVector<T>& a, t8gpu::SharedDeviceVector<T>& b) {
  t8gpu::SharedDeviceVector<T>::swap(a, b);
}
