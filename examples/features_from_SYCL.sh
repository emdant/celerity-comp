#!/bin/bash
alias sycl="clang++ -O2 -fsycl -fsycl-device-only"
alias opt=opt-15
alias llvm-dis=llvm-dis-15

# SYCL compilation
for file in sycl/*.cpp; do
  echo "SYCL source: $file"
  name="${file%.*}"

  sycl $file -o $name.bc
  llvm-dis $name.bc
done

# feature extraction from bitcode
for file in sycl/*.bc; do
    echo "--- extracing features from sycl/$file ---"
    opt -load-pass-plugin ../libfeature_pass.so  --passes="print<feature>" -disable-output $file

    # ../features -i $bc 
    # ../features -i $bc -fe kofler
    # ../features -i $bc -fs full
done
