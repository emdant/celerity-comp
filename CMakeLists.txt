cmake_minimum_required(VERSION 3.13.4)
project(Celerity_Comp)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PROJECT_SOURCE_DIR}/cmake")

# Requires LLVM
find_package(LLVM 15 REQUIRED)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Found LLVM Tools in ${LLVM_TOOLS_BINARY_DIR}")
llvm_map_components_to_libnames(llvm_libs support passes core irreader analysis)

# separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
# add_definitions(${LLVM_DEFINITIONS_LIST})

# Optional: Enable support for polynomial cost relations
option(EXTRACTOR_TOOL "Build the external feature extractor tool" OFF)

# Optional: Support more advanced and accurate feature peresentation
option(POLFEAT "Support for polynomial features" OFF)

# Optional: Install sample scripts for several
option(SAMPLE_SCRIPTS "Install sample scripts for C functions, OpenCL and SYCL" ON)

# Optional: Celerity runtime integration
option(CELERITY_RUNTIME "Install the integration layer for Celerity (requires existing Celerity Runtime installation)" OFF)

add_subdirectory(feature-pass)

# Build the feature extraction tool
if(EXTRACTOR_TOOL)
  add_subdirectory(extractor-tool)
endif()

# Build the integration layer to be used with the Celerity runtime
if(CELERITY_RUNTIME)
  add_subdirectory(celerity-pass)
endif()

if(SAMPLE_SCRIPTS)
  install(DIRECTORY "examples/" DESTINATION "${CMAKE_BINARY_DIR}/samples")
endif()
