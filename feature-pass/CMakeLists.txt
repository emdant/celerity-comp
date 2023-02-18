add_library(feature_pass SHARED)
target_include_directories(feature_pass PUBLIC include ${LLVM_INCLUDE_DIRS})

# target_link_libraries(feature_pass PUBLIC ${llvm_libs})
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
  src/KernelInvariant.cpp
)

# Support for polynomial features
if(POLFEAT)
  find_package(FLINT REQUIRED)
  target_include_directories(feature_pass PUBLIC include/polfeat ${FLINT_INCLUDE_DIRS})
  target_sources(feature_pass
    PRIVATE
    src/PolFeatAnalysis.cpp
    src/IMPoly.cpp
  )
  target_link_libraries(feature_pass PUBLIC ${FLINT_LIBRARIES})

  # IMPoly test function
  # add_executable(test_impoly ${FEATURE_SRC} ../test/test_impoly.cpp)

  # target_include_directories(test_impoly PUBLIC feature_pass)
  # target_compile_options(test_impoly PUBLIC -Wl,-znodelete)
endif()

if(SAMPLE_SCRIPTS)
  install(TARGETS feature_pass DESTINATION "${CMAKE_BINARY_DIR}/samples")
endif()