set(FANN_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/../external/fann-install/include)
set(FANN_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/../external/fann-install/lib)

set_source_files_properties("${FANN_INCLUDE_DIR}/fann.h" "${FANN_INCLUDE_DIR}/doublefann.h" "${FANN_INCLUDE_DIR}/floatfann.h"
                            PROPERTIES
                            GENERATED 1)
add_executable(main main.c)
target_link_directories(main PUBLIC ${FANN_LIBRARY_DIR})
target_include_directories(main PUBLIC ${FANN_INCLUDE_DIR})
target_link_libraries(main libfann.a libdoublefann.a libfloatfann.a m)
