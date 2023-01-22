#!/bin/bash

SYCL_FILES="example-application-sycl.cpp matrix-multiply-sycl.cpp simple-vector-add-sycl.cpp"

cd samples

# SYCL compilation
for SYCL_FILE in ${SYCL_FILES}; do
echo "SYCLFILE $SYCL_FILE"
clang++ -O2 -fsycl -fsycl-targets=spir64 -fsycl-device-only -fsycl-use-bitcode samples/simple-vector-add-sycl.cpp 
done

# feature extraction from bitcode
for bc in *.bc; do
    echo "--- extracing features from $bc ---"
    ../features -i $bc 
    ../features -i $bc -fe kofler
    ../features -i $bc -fs full
done

cd ..
