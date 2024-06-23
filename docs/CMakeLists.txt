find_package(Doxygen REQUIRED)

if(DOXYGEN_FOUND)

  set(DOXYGEN_HTML_OUTPUT ${PROJECT_BINARY_DIR}/docs/html)
  set(DOXYGEN_GENERATE_HTML YES)
  set(DOXYGEN_HAVE_DOT YES)

  doxygen_add_docs(documentation
    ${PROJECT_SOURCE_DIR}
    ALL
    COMMENT "Generate Doxygen HTML documentation")

else()
    message(FATAL_ERROR "Doxygen needs to be installed to generate the documentation")
endif()
