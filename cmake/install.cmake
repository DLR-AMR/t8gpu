install(DIRECTORY t8gpu DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(TARGETS t8gpu EXPORT ${PROJECT_NAME}-targets)

include(CMakePackageConfigHelpers)

configure_package_config_file(cmake/config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION cmake)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}ConfigVersion.cmake
  COMPATIBILITY SameMajorVersion
  VERSION "0.0.0")

install(EXPORT ${PROJECT_NAME}-targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION cmake)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION cmake)
