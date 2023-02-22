CELERITY Compiler Repository
=========

**Celerity-comp** is a collection of LLVM passes and supporting code that interact with modern predictive optimizer and runtime systems.  It is designed to work both as a standalone library, as supporting external tool, and integrated within the Celerity runtime systems. It supports the new LLVM pass manager and does not implement the legacy one.
Celerity-comp has been tested with LLVM 15. 
The LLVM passes can also interact with OpenCL and SYCL code.

### Table of contents
* [Requirements](#requirements)
  * [Common](#common)
  * [Extractor tools](#extractor)
  * [Celerity integration](#celerityintegration)
* [References](#references)

Requirements
============
  * CMake: we suggest to install both cmake and the curses gui.
  ```console
  sudo apt install cmake cmake-curses-gui
  ```
  
  * Clang/LLVM: required at least LLVM version 15.0.
  ```console
  sudo apt install clang-15 llvm-15
  ```

  * (Optional) Libflint is required for more accurate features (polfeat). Note that older versions of libflint do not support multivariate polynomials.
  ```console
  sudo apt install libflint-2.8.4 libflint-dev zlib1g-dev
  ```
  * For compatibility with SYCL, Celerity-comp has been tested with the [2022-09](https://github.com/intel/llvm/releases/tag/2022-09) release of oneAPI DPC++ compiler based on LLVM. The use of other releases may cause incompatibility issues. For getting started with DPC++ follow this [guide](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md).

Installation
============
We suggest using the cmake curses gui to select the desired features. Using `cmake` without any further option, will generate files for the library pass and for the samples.

```
mkdir build
cd build
ccmake ..
make install
```

Running the above commands will create the `feature-pass/` and  the `samples/` folders.

Getting Started
===============  
In the built `samples/` folder there will be two scripts to extract features from OpenCL and SYCL code. Some adjustments to the aliased command might be needed depending on the environment where the script is executed.

Issues
===============
The polynomial feature analysis and the extractor tool are still in development.