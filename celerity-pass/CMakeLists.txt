add_library(celerity_interface_pass SHARED src/celerity_interface_pass.cpp)
target_compile_options(celerity_interface_pass PRIVATE -Wl,-znodelete) # workoaround to fix bug with llvm autoregistring passes

if(LLVM IN_LIST LLVM_AVAILABLE_LIBS)
  target_link_libraries(celerity_interface_pass LLVM clangTooling)
else()
  target_link_libraries(celerity_interface_pass LLVMSupport clangTooling)
endif()