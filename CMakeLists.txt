cmake_minimum_required(VERSION 3.11)
project(viterbi_fann_vs_simd)

include(CheckLibraryExists)

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
ENDIF()

set(FANN_PROJECT fann_project)
set(FANN_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/fann-install)
set(FANN_LIB_DIR ${FANN_INSTALL_DIR}/lib)
set(FANN_INCLUDE_DIR ${FANN_INSTALL_DIR}/include/fann)

add_library(fixedfann STATIC IMPORTED)
set_target_properties(fixedfann PROPERTIES IMPORTED_LOCATION ${FANN_LIB_DIR}/libfixedfann.a)
target_include_directories(fixedfann INTERFACE ${FANN_INCLUDE_DIR})

add_library(doublefann STATIC IMPORTED)
set_target_properties(doublefann PROPERTIES IMPORTED_LOCATION ${FANN_LIB_DIR}/libdoublefann.a)
target_include_directories(doublefann INTERFACE ${FANN_INCLUDE_DIR})

add_library(floatfann STATIC IMPORTED)
set_target_properties(floatfann PROPERTIES IMPORTED_LOCATION ${FANN_LIB_DIR}/libfloatfann.a)
target_include_directories(floatfann INTERFACE ${FANN_INCLUDE_DIR})

add_library(fann STATIC IMPORTED)
set_target_properties(fixedfann PROPERTIES IMPORTED_LOCATION ${FANN_LIB_DIR}/libfann.a)
target_include_directories(fixedfann INTERFACE ${FANN_INCLUDE_DIR})

set(FAN_LIBRARIES fixedfann doublefann floatfann)

add_subdirectory(external)
add_subdirectory(src)
