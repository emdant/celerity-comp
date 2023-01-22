#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>

using namespace sycl;

void display_matrix(float* m, int matSize) {
  if (matSize > 16) {
    return;
  }

  std::cout << "=======" << std::endl;
  for (int i = 0; i < matSize; i++) {
    for (int j = 0; j < matSize; j++) {
      std::cout << m[i * matSize + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "=======" << std::endl;
  ;
}

void block_host(float* MA, float* MB, float* MC, int matSize) {
  int block_size = 32;
  int numBlocks = block_size / matSize;
  int extraBlockLength = block_size % matSize;
  numBlocks = extraBlockLength ? (numBlocks + 1) : (numBlocks);

#pragma omp parallel for collapse(2)
  for (int bIndexI = 0; bIndexI < matSize; bIndexI += block_size)
    for (int bIndexJ = 0; bIndexJ < matSize; bIndexJ += block_size)
      for (int bIndexK = 0; bIndexK < matSize; bIndexK += block_size) {
        int i = bIndexI;
        int j = bIndexJ;
        int k = bIndexK;
        for (int bi = i; bi < std::min(i + block_size, matSize); bi++)
          for (int bj = j; bj < std::min(j + block_size, matSize); bj++)
            for (int bk = k; bk < std::min(k + block_size, matSize); bk++) {
              MC[bi * matSize + bj] +=
                  MA[bi * matSize + bk] * MB[bk * matSize + bj];
            }
      }
}

inline int prevPowerOfTwo(int x) {
  if (x < 0) {
    return 0;
  }
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x - (x >> 1);
}

inline bool isPowerOfTwo(int x) {
  return (x & (x - 1)) == 0;
}

template <typename T>
bool local_mxm(::queue& q, T* MA, T* MB, T* MC, int matSize) {
  if (!isPowerOfTwo(matSize)) {
    std::cout << " This example only works with power of two sizes "
              << std::endl;
    return true;
  }

  auto device = q.get_device();
  auto maxBlockSize =
      device.get_info<::info::device::max_work_group_size>();
  auto blockSize = prevPowerOfTwo(std::sqrt(maxBlockSize));
  std::cout << " The Device Max Work Group Size is : " << maxBlockSize
            << std::endl;
  std::cout << " The order is : " << matSize << std::endl;
  std::cout << " The blockSize is : " << blockSize << std::endl;
  blockSize = std::min(matSize, blockSize);

  {
    range<1> dimensions(matSize * matSize);
    const property_list props = {property::buffer::use_host_ptr()};
    buffer<T> bA(MA, dimensions, props);
    buffer<T> bB(MB, dimensions, props);
    buffer<T> bC(MC, dimensions, props);

    q.submit([&](handler& cgh) {
      auto pA = bA.template get_access<access::mode::read>(cgh);
      auto pB = bB.template get_access<access::mode::read>(cgh);
      auto pC = bC.template get_access<access::mode::write>(cgh);
      auto localRange = range<1>(blockSize * blockSize);

      accessor<T, 1, access::mode::read_write, access::target::local> pBA(
          localRange, cgh);
      accessor<T, 1, access::mode::read_write, access::target::local> pBB(
          localRange, cgh);

      cgh.parallel_for<class mxm_kernel>(
          nd_range<2>{range<2>(matSize, matSize),
                      range<2>(blockSize, blockSize)},
          [=](nd_item<2> it) {
            // Current block
            int blockX = it.get_group(0);
            int blockY = it.get_group(1);

            // Current local item
            int localX = it.get_local_id(0);
            int localY = it.get_local_id(1);

            // Start in the A matrix
            int a_start = matSize * blockSize * blockY;
            // End in the b matrix
            int a_end = a_start + matSize - 1;
            // Start in the b matrix
            int b_start = blockSize * blockX;

            // Result for the current C(i,j) element
            T tmp = 0.0f;
            // We go through all a, b blocks
            for (int a = a_start, b = b_start; a <= a_end;
                 a += blockSize, b += (blockSize * matSize)) {
              // Copy the values in shared memory collectively
              pBA[localY * blockSize + localX] =
                  pA[a + matSize * localY + localX];
              // Note the swap of X/Y to maintain contiguous access
              pBB[localX * blockSize + localY] =
                  pB[b + matSize * localY + localX];
              it.barrier(access::fence_space::local_space);
              // Now each thread adds the value of its sum
              for (int k = 0; k < blockSize; k++) {
                tmp +=
                    pBA[localY * blockSize + k] * pBB[localX * blockSize + k];
              }
              // The barrier ensures that all threads have written to local
              // memory before continuing
              it.barrier(access::fence_space::local_space);
            }
            auto elemIndex =
                it.get_global_id(1) * it.get_global_range()[0] + it.get_global_id(0);
            // Each thread updates its position
            pC[elemIndex] = tmp;
          });
    });
  }
  return false;
}

/* Helper function to indicate the parameters the sample takes. */
void usage(std::string programName) {
  std::cout << " Incorrect number of parameters " << std::endl;
  std::cout << " Usage: " << std::endl;
  std::cout << programName << " [matrix size] [omp|sycl]" << std::endl;
  std::cout << "[matrix size] : Size of the matrix to multiply (minimum 32)"
            << std::endl;
  std::cout << "[omp|sycl]    : Run the OpenMP or the SYCL variant. "
            << " Default is to use both " << std::endl;
}

int main(int argc, char* argv[]) {
  float* MA;
  float* MB;
  float* MC;
  bool sycl = true;
  bool omp = true;
  bool error = false;

  if (argc != 2 && argc != 3) {
    usage(argv[0]);
    return 1;
  }

  int matSize = 0;
  try {
    matSize = std::stoi(argv[1]);
  } catch (...) {
    usage(argv[0]);
    return 1;
  }

  if (matSize < 32) {
    usage(argv[0]);
    return 1;
  }

  if (argc == 3) {
    if (std::string(argv[2]) == "omp") {
      omp = true;
      sycl = false;
    } else if (std::string(argv[2]) == "sycl") {
      omp = false;
      sycl = true;
    } else {
      usage(argv[0]);
    }
  }

  MA = new float[matSize * matSize];
  MB = new float[matSize * matSize];
  MC = new float[matSize * matSize];

// Matrix initialization
#pragma omp parallel for collapse(2)
  for (int i = 0; i < matSize; i++)
    for (int j = 0; j < matSize; j++) {
      MA[i * matSize + j] = 0.0f;
      if (i == j) {
        MA[i * matSize + j] = 1.0f;
      }
      MB[i * matSize + j] = 2.0f;
      MC[i * matSize + j] = 0.0f;  // i * matSize + j;
    }

  std::cout << " Input matrix " << std::endl;
  display_matrix(MA, matSize);
  display_matrix(MB, matSize);
  display_matrix(MC, matSize);

  if (omp) {
#if defined(_OPENMP)
    std::cout << "OpenMP: ";
#else
    std::cout << "C++: ";
#endif

    {
      auto start = std::chrono::steady_clock::now();
      block_host(MA, MB, MC, matSize);
      auto end = std::chrono::steady_clock::now();
      auto time =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      std::cout << "Time: " << time << std::endl;
      float flops =
          (2.0f * matSize * matSize * matSize / (time / 1000.0f)) * 1.0e-9f;
      std::cout << "GFLOPs: " << flops << std::endl;

      bool error = false;
      // Testing
      for (int i = 0; i < matSize; i++)
        for (int j = 0; j < matSize; j++) {
          if (std::fabs(MC[i * matSize + j] - MB[i * matSize + j]) > 1e-8) {
            std::cout << " Position " << i << ", " << j
                      << " differs: " << MC[i * matSize + j]
                      << " != " << MB[i * matSize + j] << std::endl;
            error = true;
          }
        }
      if (!error) {
        std::cout << "Success" << std::endl;
      } else {
        std::cout << " Error in the computation " << std::endl;
      }
    }
  }

  if (sycl) {
    std::cout << " ***** SYCL " << std::endl;
    // Matrix initialization
    for (int i = 0; i < matSize; i++)
      for (int j = 0; j < matSize; j++) {
        MC[i * matSize + j] = 0.0f;  // i * matSize + j;
      }

    {
      {
        queue q([&](exception_list eL) {
          try {
            for (auto& e : eL) {
              std::rethrow_exception(e);
            }
          } catch (::exception e) {
            std::cout << " An exception has been thrown: " << e.what()
                      << std::endl;
          }
        });

        auto start = std::chrono::steady_clock::now();
        error = local_mxm(q, MA, MB, MC, matSize);
        q.wait_and_throw();
        auto end = std::chrono::steady_clock::now();
        auto time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
        std::cout << "SYCL: ";
        std::cout << "Time: " << time << std::endl;
        float flops =
            (2.0f * matSize * matSize * matSize / (time / 1000.0f)) * 1.0e-9f;
        std::cout << "GFLOPs: " << flops << std::endl;
        std::cout << " Output " << std::endl;
      }

      if (!error) {
        display_matrix(MC, matSize);
        error = false;
        // Testing
        for (int i = 0; i < matSize; i++)
          for (int j = 0; j < matSize; j++) {
            if (std::fabs(MC[i * matSize + j] - MB[i * matSize + j]) > 1e-8) {
              std::cout << " Position " << i << ", " << j
                        << " differs: " << MC[i * matSize + j]
                        << " != " << MB[i * matSize + j] << std::endl;
              error = true;
            }
          }
        if (!error) {
          std::cout << "Success" << std::endl;
          ;
        } else {
          std::cout << " Error in the computation " << std::endl;
        }
      }
    }
  }

  delete[] MA;
  delete[] MB;
  delete[] MC;

  return error ? 1 : 0;
}
