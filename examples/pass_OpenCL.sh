#!/bin/bash
alias clang=clang-15
alias opt="opt-15 -load-pass-plugin ../libfeature_pass.so  --passes=\"print<feature>\" -disable-output"

echo "$(tput setaf 1) Generating LLVM IR from generic C function and OpenCL kernels $(tput sgr 0)"
for file in opencl/*.cl; do
  name="${file%.*}"
  clang -c -x cl -emit-llvm -cl-std=CL2.0 -Xclang -finclude-default-header $file -o $name.bc
  llvm-dis $name.bc
done

for file in opencl/*.c; do
  name="${file%.*}"
  clang -c -emit-llvm $file -o $name.bc
  llvm-dis $name.bc
done

echo "$(tput setaf 1) Feature evaluation from LLVM IR with LLVM modular optimizer (opt) $(tput sgr 0)"
for file in opencl/*.bc; do
  opt $file
done

# echo
# echo "$(tput setaf 1) Feature extraction from LLVM IR with the extractor utility $(tput sgr 0)"
# ./feature_ext -i samples/vecadd.bc
# ./feature_ext -i samples/vecadd.bc -fe kofler13   

