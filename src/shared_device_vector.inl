#include <shared_device_vector.h>

/// default implementation
template<typename T>
inline t8gpu::SharedDeviceVector<T>::SharedDeviceVector(size_t size, sc_MPI_Comm comm)
  : m_size {size},
    m_comm {comm},
    m_capacity {size} {
  MPI_Comm_size(m_comm, &m_nb_ranks);
  MPI_Comm_rank(m_comm, &m_rank);

  m_handles.resize(m_nb_ranks);
  m_arrays.resize(m_nb_ranks);
  thrust::fill(m_arrays.begin(), m_arrays.end(), nullptr);

  if (size > 0) {
    T8GPU_CUDA_CHECK_ERROR(cudaMalloc(&m_arrays[m_rank], sizeof(T)*m_size));
    T8GPU_CUDA_CHECK_ERROR(cudaIpcGetMemHandle(&(m_handles[m_rank].handle), m_arrays[m_rank]));
    m_handles[m_rank].need_open_ipc = true;
  } else {
    m_handles[m_rank].need_open_ipc = false;
  }
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, m_handles.data(), sizeof(SharedDeviceVector<T>::Handle), MPI_BYTE, m_comm);

  for (int i=0; i<m_nb_ranks; i++) {
    if (i != m_rank && m_handles[i].need_open_ipc) {
      T8GPU_CUDA_CHECK_ERROR(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&m_arrays[i]), m_handles[i].handle, cudaIpcMemLazyEnablePeerAccess));
    }
  }
  m_device_arrays = m_arrays;
}

template<typename T>
inline t8gpu::SharedDeviceVector<T>::~SharedDeviceVector() {
  for (int i=0; i<m_nb_ranks; i++) {
    if (m_arrays[i] != nullptr) {
      if (i == m_rank) {
	T8GPU_CUDA_CHECK_ERROR(cudaFree(m_arrays[i]));
      } else {
	T8GPU_CUDA_CHECK_ERROR(cudaIpcCloseMemHandle(m_arrays[i]));
      }
    }
  }
}

template<typename T>
inline t8gpu::SharedDeviceVector<T>::SharedDeviceVector(SharedDeviceVector<T>&& other)
  : m_rank {other.m_rank},
    m_nb_ranks {other.m_nb_ranks},
    m_size {other.m_size},
    m_capacity {other.m_capacity},
    m_handles {std::move(other.m_handles)},
    m_arrays {other.m_arrays},
    m_device_arrays {std::move(other.m_device_arrays)} {
  for (int i=0; i<m_nb_ranks; i++) {
    other.m_arrays[i] = nullptr;
  }
}

template<typename T>
inline t8gpu::SharedDeviceVector<T>& t8gpu::SharedDeviceVector<T>::operator=(SharedDeviceVector&& other) {
  this->~SharedDeviceVector();
  m_rank = other.m_rank;
  m_nb_ranks = other.m_nb_ranks;
  m_size = other.m_size;
  m_capacity = other.m_capacity;
  m_handles = std::move(other.m_handles);
  m_arrays = other.m_arrays;
  m_device_arrays = std::move(other.m_device_arrays);

  for (int i=0; i<m_nb_ranks; i++) {
    other.m_arrays[i] = nullptr;
  }
  return *this;
}

template<typename T>
inline void t8gpu::SharedDeviceVector<T>::resize(size_t new_size) {
  if (new_size <= m_capacity) {
    m_size = new_size;
    m_handles[m_rank].need_open_ipc = false;
  } else {
    T8GPU_CUDA_CHECK_ERROR(cudaFree(m_arrays[m_rank]));

    m_capacity = new_size + new_size / 2;
    m_size = new_size;

    T* new_allocation {};
    T8GPU_CUDA_CHECK_ERROR(cudaMalloc(&new_allocation, sizeof(T)*m_capacity));
    m_arrays[m_rank] = new_allocation;

    T8GPU_CUDA_CHECK_ERROR(cudaIpcGetMemHandle(&(m_handles[m_rank].handle), m_arrays[m_rank]));
    m_handles[m_rank].need_open_ipc = true;
  }
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, m_handles.data(), sizeof(SharedDeviceVector<T>::Handle), MPI_BYTE, m_comm);
  for (int i=0; i<m_nb_ranks; i++) {
    if (i != m_rank && m_handles[i].need_open_ipc) {
      if (m_arrays[i] != nullptr) {
	T8GPU_CUDA_CHECK_ERROR(cudaIpcCloseMemHandle(m_arrays[i]));
      }
      T8GPU_CUDA_CHECK_ERROR(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&m_arrays[i]), m_handles[i].handle, cudaIpcMemLazyEnablePeerAccess));
    }
  }
  m_device_arrays = m_arrays;
}

template<typename T>
inline const t8gpu::SharedDeviceVector<T>& t8gpu::SharedDeviceVector<T>::operator=(const thrust::host_vector<T>& other) {
  this->resize(other.size());
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(m_arrays[m_rank], thrust::raw_pointer_cast(other.data()), sizeof(T)*m_size, cudaMemcpyHostToDevice));
  return *this;
}

template<typename T>
inline const t8gpu::SharedDeviceVector<T>& t8gpu::SharedDeviceVector<T>::operator=(const thrust::device_vector<T>& other) {
  this->resize(other.size());
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(m_arrays[m_rank], thrust::raw_pointer_cast(other.data()), sizeof(T)*m_size, cudaMemcpyDeviceToDevice));
  return *this;
}
template<typename T>
[[nodiscard]] inline size_t t8gpu::SharedDeviceVector<T>::size() const { return m_size; }

template<typename T>
inline void t8gpu::SharedDeviceVector<T>::clear() { m_size = 0; }

template<typename T>
[[nodiscard]] inline T* t8gpu::SharedDeviceVector<T>::get_own() { return m_arrays[m_rank]; }

template<typename T>
[[nodiscard]] inline T** t8gpu::SharedDeviceVector<T>::get_all() { return thrust::raw_pointer_cast(m_device_arrays.data()); }

template<typename T>
[[nodiscard]] inline T const* t8gpu::SharedDeviceVector<T>::get_own() const { return m_arrays[m_rank]; }

template<typename T>
[[nodiscard]] inline T const* const* t8gpu::SharedDeviceVector<T>::get_all() const { return thrust::raw_pointer_cast(m_device_arrays.data()); }

// Specialization for std::array<T, N>
template<typename T, size_t N>
inline t8gpu::SharedDeviceVector<std::array<T, N>>::SharedDeviceVector(size_t size, sc_MPI_Comm comm)
  : m_size {size},
    m_comm {comm},
    m_capacity {size} {
  MPI_Comm_size(m_comm, &m_nb_ranks);
  MPI_Comm_rank(m_comm, &m_rank);

  m_handles.resize(m_nb_ranks);
  m_allocations.resize(m_nb_ranks);
  m_arrays.resize(m_nb_ranks * N);
  thrust::fill(m_allocations.begin(), m_allocations.end(), nullptr);
  thrust::fill(m_arrays.begin(), m_arrays.end(), nullptr);

  if (size > 0) {
    T8GPU_CUDA_CHECK_ERROR(cudaMalloc(&m_allocations[m_rank], sizeof(T)*m_capacity* N));
    T8GPU_CUDA_CHECK_ERROR(cudaIpcGetMemHandle(&(m_handles[m_rank].handle), m_allocations[m_rank]));
    m_handles[m_rank].need_open_ipc = true;
    m_handles[m_rank].capacity = m_capacity;
  } else {
    m_handles[m_rank].need_open_ipc = false;
    m_handles[m_rank].capacity = m_capacity;
  }
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, m_handles.data(), sizeof(SharedDeviceVector<std::array<T, N>>::Handle), MPI_BYTE, m_comm);

  for (int i=0; i<m_nb_ranks; i++) {
    if (i != m_rank && m_handles[i].need_open_ipc) {
      T8GPU_CUDA_CHECK_ERROR(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&m_allocations[i]), m_handles[i].handle, cudaIpcMemLazyEnablePeerAccess));
    }
  }
  for (size_t i=0; i<N; i++) {
    for (int j=0; j<m_nb_ranks; j++) {
      m_arrays[i*m_nb_ranks+j] = m_allocations[j] + i*m_handles[j].capacity;
    }
  }
  m_device_arrays = m_arrays;
}

template<typename T, size_t N>
inline t8gpu::SharedDeviceVector<std::array<T, N>>::~SharedDeviceVector() {
  for (int i=0; i<m_nb_ranks; i++) {
    if (m_allocations[i] != nullptr) {
      if (i == m_rank) {
	T8GPU_CUDA_CHECK_ERROR(cudaFree(m_allocations[i]));
      } else {
	T8GPU_CUDA_CHECK_ERROR(cudaIpcCloseMemHandle(m_allocations[i]));
      }
    }
  }
}

template<typename T, size_t N>
inline t8gpu::SharedDeviceVector<std::array<T, N>>::SharedDeviceVector(SharedDeviceVector<std::array<T, N>>&& other)
  : m_rank {other.m_rank},
    m_nb_ranks {other.m_nb_ranks},
    m_size {other.m_size},
    m_capacity {other.m_capacity},
    m_handles {std::move(other.m_handles)},
    m_allocations {other.m_allocations},
    m_arrays {other.m_arrays},
    m_device_arrays {std::move(other.m_device_arrays)} {
  for (int i=0; i<m_nb_ranks; i++) {
    other.m_allocations[i] = nullptr;
  }
}

template<typename T, size_t N>
inline t8gpu::SharedDeviceVector<std::array<T, N>>& t8gpu::SharedDeviceVector<std::array<T, N>>::operator=(SharedDeviceVector&& other) {
  this->~SharedDeviceVector();
  m_rank = other.m_rank;
  m_nb_ranks = other.m_nb_ranks;
  m_size = other.m_size;
  m_capacity = other.m_capacity;
  m_handles = std::move(other.m_handles);
  m_arrays = other.m_arrays;
  m_allocations = other.m_allocations;
  m_device_arrays = std::move(other.m_device_arrays);

  for (int i=0; i<m_nb_ranks; i++) {
    other.m_allocations[i] = nullptr;
  }
  return *this;
}

template<typename T, size_t N>
inline void t8gpu::SharedDeviceVector<std::array<T, N>>::resize(size_t new_size) {
  if (new_size <= m_capacity) {
    m_size = new_size;
    m_handles[m_rank].need_open_ipc = false;
  } else {
    T8GPU_CUDA_CHECK_ERROR(cudaFree(m_allocations[m_rank]));

    m_capacity = new_size + new_size / 2;
    m_size = new_size;

    T* new_allocation {};
    T8GPU_CUDA_CHECK_ERROR(cudaMalloc(&new_allocation, sizeof(T)*m_capacity * N));
    m_allocations[m_rank] = new_allocation;
    for (size_t j=0; j<N; j++) {
      m_arrays[j*m_nb_ranks+m_rank] = m_allocations[m_rank] + j*m_capacity;
    }

    T8GPU_CUDA_CHECK_ERROR(cudaIpcGetMemHandle(&(m_handles[m_rank].handle), m_allocations[m_rank]));
    m_handles[m_rank].need_open_ipc = true;
    m_handles[m_rank].capacity = m_capacity;
  }
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, m_handles.data(), sizeof(SharedDeviceVector<std::array<T, N>>::Handle), MPI_BYTE, m_comm);
  for (int i=0; i<m_nb_ranks; i++) {
    if (i != m_rank && m_handles[i].need_open_ipc) {
      if (m_allocations[i] != nullptr) {
	T8GPU_CUDA_CHECK_ERROR(cudaIpcCloseMemHandle(m_allocations[i]));
      }
      T8GPU_CUDA_CHECK_ERROR(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&m_allocations[i]), m_handles[i].handle, cudaIpcMemLazyEnablePeerAccess));
      for (size_t j=0; j<N; j++) {
	m_arrays[j*m_nb_ranks+i] = m_allocations[i] + j*m_handles[i].capacity;
      }
    }
  }
  m_device_arrays = m_arrays;
}

template<typename T, size_t N>
inline void t8gpu::SharedDeviceVector<std::array<T, N>>::copy(size_t index, const thrust::host_vector<T>& vector) {
  assert(m_size <= vector.size());
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(m_arrays[index*m_nb_ranks+m_rank], thrust::raw_pointer_cast(vector.data()), sizeof(T)*vector.size(), cudaMemcpyHostToDevice));
}

template<typename T, size_t N>
inline void t8gpu::SharedDeviceVector<std::array<T, N>>::copy(size_t index, const thrust::device_vector<T>& vector) {
  assert(m_size <= vector.size());
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(m_arrays[index*m_nb_ranks+m_rank], thrust::raw_pointer_cast(vector.data()), sizeof(T)*vector.size(), cudaMemcpyDeviceToDevice));
}

template<typename T, size_t N>
inline void t8gpu::SharedDeviceVector<std::array<T, N>>::copy(size_t index, T const* buffer, size_t num_elements) {
  assert(num_elements <= m_size);
  T8GPU_CUDA_CHECK_ERROR(cudaMemcpy(m_arrays[index*m_nb_ranks+m_rank], buffer, sizeof(T)*num_elements, cudaMemcpyDeviceToDevice));
}

template<typename T, size_t N>
[[nodiscard]] inline size_t t8gpu::SharedDeviceVector<std::array<T, N>>::size() const { return m_size; }

template<typename T, size_t N>
inline void t8gpu::SharedDeviceVector<std::array<T, N>>::clear() { m_size = 0; }

template<typename T, size_t N>
[[nodiscard]] inline T* t8gpu::SharedDeviceVector<std::array<T, N>>::get_own(int index) { return m_arrays[index*m_nb_ranks+m_rank]; }

template<typename T, size_t N>
[[nodiscard]] inline T** t8gpu::SharedDeviceVector<std::array<T, N>>::get_all(int index) { return thrust::raw_pointer_cast(m_device_arrays.data() + index*m_nb_ranks); }

template<typename T, size_t N>
[[nodiscard]] inline T const* t8gpu::SharedDeviceVector<std::array<T, N>>::get_own(int index) const { return m_arrays[index*m_nb_ranks+m_rank]; }

template<typename T, size_t N>
[[nodiscard]] inline T const* const* t8gpu::SharedDeviceVector<std::array<T, N>>::get_all(int index) const { return thrust::raw_pointer_cast(m_device_arrays.data() + index*m_nb_ranks); }
