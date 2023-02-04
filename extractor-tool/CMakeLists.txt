add_executable(extractor_tool FeatureTool.cpp ExtractorCommandLine.cpp)
target_include_directories(extractor_tool PUBLIC ${LLVM_INCLUDE_DIRS})
target_link_libraries(extractor_tool PUBLIC ${llvm_libs})
target_compile_options(extractor_tool PUBLIC -Wl,-znodelete)

if(SAMPLE_SCRIPTS)
  install(TARGETS extractor_tool DESTINATION "${CMAKE_BINARY_DIR}/samples")
endif()