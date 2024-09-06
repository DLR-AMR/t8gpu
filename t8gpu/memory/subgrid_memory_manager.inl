#include "subgrid_memory_manager.h"

template<typename VariableType, typename StepType, typename SubgridType>
t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::SubgridMemoryManager(size_t      nb_elements,
                                                                                       sc_MPI_Comm comm)
    : m_device_buffer(nb_elements * SubgridType::size, comm), m_device_volume(nb_elements, comm) {}

template<typename VariableType, typename StepType, typename SubgridType>
void t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::resize(size_t new_size) {
  m_device_buffer.resize(new_size * SubgridType::size);
  m_device_buffer.resize(new_size);
}

template<typename VariableType, typename StepType, typename SubgridType>
void t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::set_variable(
    step_index_type step, variable_index_type variable, thrust::device_vector<float_type> const& buffer) {
  m_device_buffer.copy(step * nb_variables + variable, buffer);
}

template<typename VariableType, typename StepType, typename SubgridType>
void t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::set_variable(
    step_index_type step, variable_index_type variable, thrust::host_vector<float_type> const& buffer) {
  m_device_buffer.copy(step * nb_variables + variable, buffer);
}

template<typename VariableType, typename StepType, typename SubgridType>
void t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::set_variable(step_index_type     step,
                                                                                    variable_index_type variable,
                                                                                    float_type*         buffer) {
  m_device_buffer.copy(step * nb_variables + variable, buffer, m_device_buffer.size() * SubgridType::size);
}

template<typename VariableType, typename StepType, typename SubgridType>
void t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::set_volume(
    thrust::host_vector<float_type> const& buffer) {
  m_device_volume = buffer;
}

template<typename VariableType, typename StepType, typename SubgridType>
void t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::set_volume(
    thrust::device_vector<float_type> const& buffer) {
  m_device_volume = buffer;
}

template<typename VariableType, typename StepType, typename SubgridType>
t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::float_type*
t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::get_own_volume() {
  return m_device_volume.get_own();
}

template<typename VariableType, typename StepType, typename SubgridType>
t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::float_type const*
t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::get_own_volume() const {
  return m_device_volume.get_own();
}

template<typename VariableType, typename StepType, typename SubgridType>
t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::float_type* const*
t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::get_all_volume() {
  return m_device_volume.get_all();
}

template<typename VariableType, typename StepType, typename SubgridType>
t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::float_type const* const*
t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::get_all_volume() const {
  return m_device_volume.get_all();
}

template<typename VariableType, typename StepType, typename SubgridType>
t8gpu::SubgridMemoryAccessorOwn<VariableType, SubgridType>
t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::get_own_variables(
    t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::step_index_type step) {
  std::array<float_type*, nb_variables> array{};
  for (int k = 0; k < static_cast<int>(nb_variables); k++) {
    array[k] = m_device_buffer.get_own(step * static_cast<int>(nb_variables) + k);
  }
  return SubgridMemoryAccessorOwn<VariableType, SubgridType>{array};
}

template<typename VariableType, typename StepType, typename SubgridType>
t8gpu::SubgridMemoryAccessorAll<VariableType, SubgridType>
t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::get_all_variables(
    t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::step_index_type step) {
  std::array<float_type* const*, nb_variables> array{};
  for (int k = 0; k < static_cast<int>(nb_variables); k++) {
    array[k] = m_device_buffer.get_all(step * static_cast<int>(nb_variables) + k);
  }
  return SubgridMemoryAccessorAll<VariableType, SubgridType>{array};
}

template<typename VariableType, typename StepType, typename SubgridType>
typename SubgridType::accessor_type<t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::float_type>
t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::get_own_variable(
    t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::step_index_type     step,
    t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::variable_index_type variable) {
  return typename SubgridType::accessor_type<float_type>{
      m_device_buffer.get_own(step * static_cast<int>(nb_variables) + static_cast<int>(variable))};
}

template<typename VariableType, typename StepType, typename SubgridType>
typename SubgridType::accessor_type<t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::float_type> const
t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::get_own_variable(
    t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::step_index_type     step,
    t8gpu::SubgridMemoryManager<VariableType, StepType, SubgridType>::variable_index_type variable) const {
  return typename SubgridType::accessor_type<float_type>{
      m_device_buffer.get_own(step * static_cast<int>(nb_variables) + static_cast<int>(variable))};
}
