#include <memory_manager.h>

template<typename VariableType, typename StepType>
t8gpu::MemoryManager<VariableType, StepType>::MemoryManager(size_t nb_elements, sc_MPI_Comm comm) : m_device_buffer(nb_elements, comm) {}

template<typename VariableType, typename StepType>
void t8gpu::MemoryManager<VariableType, StepType>::resize(size_t new_size) {
  m_device_buffer.resize(new_size);
}

template<typename VariableType, typename StepType>
void t8gpu::MemoryManager<VariableType, StepType>::set_variable(step_index_type step, variable_index_type variable, const thrust::device_vector<float_type>& buffer) {
  m_device_buffer.copy(step*nb_variables + variable, buffer);
}

template<typename VariableType, typename StepType>
void t8gpu::MemoryManager<VariableType, StepType>::set_variable(step_index_type step, variable_index_type variable, const thrust::host_vector<float_type>& buffer) {
  m_device_buffer.copy(step*nb_variables + variable, buffer);
}

template<typename VariableType, typename StepType>
void t8gpu::MemoryManager<VariableType, StepType>::set_variable(step_index_type step, variable_index_type variable, float_type* buffer) {
  m_device_buffer.copy(step*nb_variables + variable, buffer, m_device_buffer.size());
}

template<typename VariableType, typename StepType>
void t8gpu::MemoryManager<VariableType, StepType>::set_volume(const thrust::host_vector<float_type>& buffer) {
  m_device_buffer.copy(nb_steps*nb_variables, buffer);
}

template<typename VariableType, typename StepType>
void t8gpu::MemoryManager<VariableType, StepType>::set_volume(const thrust::device_vector<float_type>& buffer) {
  m_device_buffer.copy(nb_steps*nb_variables, buffer);
}

template<typename VariableType, typename StepType>
void t8gpu::MemoryManager<VariableType, StepType>::set_volume(float_type* buffer) {
  m_device_buffer.copy(nb_steps*nb_variables, buffer, m_device_buffer.size());
}

template<typename VariableType, typename StepType>
t8gpu::MemoryManager<VariableType, StepType>::float_type* t8gpu::MemoryManager<VariableType, StepType>::get_own_volume() {
  return m_device_buffer.get_own(nb_steps*nb_variables);
}

template<typename VariableType, typename StepType>
t8gpu::MemoryManager<VariableType, StepType>::float_type const* t8gpu::MemoryManager<VariableType, StepType>::get_own_volume() const {
  return m_device_buffer.get_own(nb_steps*nb_variables);
}

template<typename VariableType, typename StepType>
t8gpu::MemoryManager<VariableType, StepType>::float_type* const* t8gpu::MemoryManager<VariableType, StepType>::get_all_volume() {
  return m_device_buffer.get_all(nb_steps*nb_variables);
}

template<typename VariableType, typename StepType>
t8gpu::MemoryManager<VariableType, StepType>::float_type const* const* t8gpu::MemoryManager<VariableType, StepType>::get_all_volume() const {
  return m_device_buffer.get_all(nb_steps*nb_variables);
}

template<typename VariableType, typename StepType>
t8gpu::MemoryAccessorOwn<VariableType> t8gpu::MemoryManager<VariableType, StepType>::get_own_variables(t8gpu::MemoryManager<VariableType, StepType>::step_index_type step) {
  std::array<float_type*, nb_variables> array {};
  for (int k=0; k<static_cast<int>(nb_variables); k++) {
    array[k] = m_device_buffer.get_own(step*static_cast<int>(nb_variables) + k);
  }
  return MemoryAccessorOwn<VariableType> {array};
}

template<typename VariableType, typename StepType>
t8gpu::MemoryAccessorAll<VariableType> t8gpu::MemoryManager<VariableType, StepType>::get_all_variables(t8gpu::MemoryManager<VariableType, StepType>::step_index_type step) {
  std::array<float_type* const*, nb_variables> array {};
  for (int k=0; k<static_cast<int>(nb_variables); k++) {
    array[k] = m_device_buffer.get_all(step*static_cast<int>(nb_variables) + k);
  }
  return MemoryAccessorAll<VariableType> {array};
}

template<typename VariableType, typename StepType>
t8gpu::MemoryManager<VariableType, StepType>::float_type* t8gpu::MemoryManager<VariableType, StepType>::get_own_variable(t8gpu::MemoryManager<VariableType, StepType>::step_index_type step, t8gpu::MemoryManager<VariableType, StepType>::variable_index_type variable) {
  return m_device_buffer.get_own(step*static_cast<int>(nb_variables) + static_cast<int>(variable));
}

template<typename VariableType, typename StepType>
t8gpu::MemoryManager<VariableType, StepType>::float_type const* t8gpu::MemoryManager<VariableType, StepType>::get_own_variable(t8gpu::MemoryManager<VariableType, StepType>::step_index_type step, t8gpu::MemoryManager<VariableType, StepType>::variable_index_type variable) const {
  return m_device_buffer.get_own(step*static_cast<int>(nb_variables) + static_cast<int>(variable));
}
