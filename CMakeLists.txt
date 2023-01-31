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
option(EXTRACTOR_TOOL "Build the external feature extractor tool" ON)

# Optional: Support more advanced and accurate feature peresentation
option(POLFEAT "Support for polynomial features" ON)

# Optional: Install sample scripts for several
option(SAMPLE_SCRIPTS "Install sample scripts for C functions, OpenCL and SYCL" ON)

# Optional: Celerity runtime integration
option(CELERITY_RUNTIME "Install the integration layer for Celerity (requires existing Celerity Runtime installation)" OFF)

# Build the LLVM pass to be used with the optimizer
add_library(feature_pass SHARED)
target_include_directories(feature_pass PUBLIC include ${LLVM_INCLUDE_DIRS})

target_sources(feature_pass
  PRIVATE
  src/feature-sets/FeatureSet.cpp
  src/feature-sets/FanFeatureSet.cpp
  src/feature-sets/Grewe11FeatureSet.cpp
  src/feature-sets/FullFeatureSet.cpp
  src/FeatureAnalysisPlugin.cpp
  src/FeatureAnalysis.cpp
  src/Kofler13Analysis.cpp
  src/DefaultFeatureAnalysis.cpp
  src/AnalysisUtils.cpp
)

# Support for polynomial features
if(POLFEAT)
  find_package(FLINT REQUIRED)
  target_include_directories(feature_pass PUBLIC ${FLINT_INCLUDE_DIRS})
  target_sources(feature_pass PRIVATE src/KernelInvariant.cpp src/PolFeatAnalysis.cpp src/IMPoly.cpp)
  target_link_libraries(feature_pass PUBLIC ${FLINT_LIBRARIES})

  # IMPoly test function
  add_executable(test_impoly ${FEATURE_SRC} test/test_impoly.cpp)

  target_link_libraries(test_impoly PUBLIC feature_pass ${llvm_libs} ${EXTRA_LIB})
  target_compile_options(test_impoly PUBLIC -Wl,-znodelete)
endif(POLFEAT)

# Build the feature extraction tool
if(EXTRACTOR_TOOL)
  add_executable(feature_ext ${FEATURE_SRC} src/feature_tool.cpp)
  target_link_libraries(feature_ext ${llvm_libs} ${EXTRA_LIB})
  target_compile_options(feature_ext PUBLIC -Wl,-znodelete)

  # target_include_directories(feature_ext ${LLVM_INCLUDE_DIRS} ${FLINT_INCLUDE_DIR} "${PROJECT_SOURCE_DIR}/include")
endif(EXTRACTOR_TOOL)

# Build the integration layer to be used with the Celerity runtime
if(CELERITY_RUNTIME)
  # build a Celerity  kernel name pass
  add_library(celerity_interface_pass SHARED src/celerity_interface_pass.cpp)
  target_compile_options(celerity_interface_pass PRIVATE -Wl,-znodelete) # workoaround to fix bug with llvm autoregistring passes

  if(LLVM IN_LIST LLVM_AVAILABLE_LIBS)
    target_link_libraries(celerity_interface_pass LLVM clangTooling)
  else()
    target_link_libraries(celerity_interface_pass LLVMSupport clangTooling)
  endif()
endif(CELERITY_RUNTIME)

# Install example scripts (make install)
if(SAMPLE_SCRIPTS)
  install(DIRECTORY "examples/" DESTINATION "${CMAKE_BINARY_DIR}/samples")
endif(SAMPLE_SCRIPTS)
