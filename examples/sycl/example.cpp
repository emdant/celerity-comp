#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;

constexpr size_t N = 100;
constexpr size_t M = 150;

int main() {
  {
    queue myQueue;

    buffer<float, 2> a(range<2>{N, M});
    buffer<float, 2> b(range<2>{N, M});
    buffer<float, 2> c(range<2>{N, M});

    myQueue.submit([&](handler& cgh) {
      auto A = a.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class InitA>(range<2>{N, M}, [=](id<2> index) {
        A[index] = index[0] * 2 + index[1];
      });
    });

    myQueue.submit([&](handler& cgh) {
      auto B = b.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class InitB>(range<2>{N, M}, [=](id<2> index) {
        B[index] = index[0] * 2014 + index[1] * 42;
      });
    });

    myQueue.submit([&](handler& cgh) {
      auto A = a.get_access<access::mode::read>(cgh);
      auto B = b.get_access<access::mode::read>(cgh);
      auto C = c.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class MatrixAdd>(
          range<2>{N, M}, [=](id<2> index) { C[index] = A[index] + B[index]; });
    });

    auto C = c.get_access<access::mode::read>();
    std::cout << "Result:" << std::endl;
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < M; j++) {
        if (C[i][j] != i * (2 + 2014) + j * (1 + 42)) {
          std::cout << "Wrong value " << C[i][j] << " for element " << i << " "
                    << j << std::endl;
          return -1;
        }
      }
    }
  }

  std::cout << "Good computation!" << std::endl;
  return 0;
}
