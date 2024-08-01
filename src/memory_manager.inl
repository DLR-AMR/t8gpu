#include <memory_manager.h>

template<typename VariableType, typename StepType>
t8gpu::MemoryManager<VariableType, StepType>::MemoryManager(size_t nb_elements) : m_device_buffer(nb_elements) {}

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
  m_device_buffer.copy(step*nb_variables + variable, buffer);
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
t8gpu::MemoryManager<VariableType, StepType>::MemoryAccessorOwn t8gpu::MemoryManager<VariableType, StepType>::get_own_variables(t8gpu::MemoryManager<VariableType, StepType>::step_index_type step) {
  MemoryAccessorOwn memory_accessor {};
  for (int k=0; k<static_cast<int>(nb_variables); k++) {
    memory_accessor.pointers[k] = m_device_buffer.get_own(step*static_cast<int>(nb_variables) + k);
  }
  return memory_accessor;
}

template<typename VariableType, typename StepType>
t8gpu::MemoryManager<VariableType, StepType>::MemoryAccessorAll t8gpu::MemoryManager<VariableType, StepType>::get_all_variables(t8gpu::MemoryManager<VariableType, StepType>::step_index_type step) {
  MemoryAccessorOwn memory_accessor {};
  for (int k=0; k<static_cast<int>(nb_variables); k++) {
    memory_accessor.pointers[k] = m_device_buffer.get_all(step*static_cast<int>(nb_variables) + k);
  }
  return memory_accessor;
}
